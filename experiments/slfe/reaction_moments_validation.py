"""Dense GPU ambient oracles for complete-state reaction and moment projection.

Distributed under the Boost Software License, Version 1.0.
The joint Gram matrices independently check positivity, coercivity, weighted
cross adjoints, constant-weight limits, and the projection Pythagorean identity.
"""

import argparse
from copy import copy
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from carrier import Carrier
from memory_transport import MemoryTransport
from reaction_moments import ReactionMoments
from campaign import provenance


def joint_matrix(coupling, operator=None, power=1):
    space = coupling.space
    nv, nw = space.mesh.cells * space.nb, coupling.n**2 * coupling.nc
    vmass = cp.repeat(space.mesh.areas, space.nb)
    wmass = cp.broadcast_to(coupling.mass, (coupling.n**2, coupling.nc)).ravel()
    vi = cp.ascontiguousarray(
        cp.eye(nv).reshape(space.mesh.cells, space.nb, nv).transpose(0, 2, 1)
    )
    projection = coupling if operator is None else operator.cross[power]
    cross = projection.project(vi).transpose(0, 2, 1).reshape(nw, nv)
    cross *= cp.sqrt(wmass[:, None] / vmass[None, :])
    if operator is None:
        fe, auxiliary = cp.eye(nv), cp.eye(nw)
    else:
        fe = operator.apply_fe(vi, power).transpose(0, 2, 1).reshape(nv, nv)
        wi = cp.ascontiguousarray(
            cp.eye(nw).reshape(coupling.n**2, coupling.nc, nw).transpose(0, 2, 1)
        )
        auxiliary = (
            operator.apply_auxiliary(wi, power).transpose(0, 2, 1).reshape(nw, nw)
        )
        auxiliary *= cp.sqrt(wmass[:, None] / wmass[None, :])
    return cp.concatenate(
        (
            cp.concatenate((fe, cross.T), axis=1),
            cp.concatenate((cross, auxiliary), axis=1),
        )
    )


def coordinates(state):
    c = state.coupling
    d = state.auxiliary - c.project(state.anchor)
    a = (
        (state.anchor * cp.sqrt(c.space.mesh.areas)[:, None, None])
        .transpose(0, 2, 1)
        .reshape(-1, state.value.shape[1])
    )
    mass = cp.broadcast_to(c.mass, (c.n**2, c.nc))
    d = (
        (d * cp.sqrt(mass)[:, None, :])
        .transpose(0, 2, 1)
        .reshape(-1, state.value.shape[1])
    )
    return cp.concatenate((a, d))


def validate(degree, krylov_seed="moment"):
    space = Space(Mesh(4, jitter=0.18, seed=817), degree)
    carrier = Carrier(space, n=4)
    rng = cp.random.RandomState(131)
    original = MemoryTransport(carrier, cp.zeros((space.mesh.cells, 2, space.nb)))
    original.anchor = rng.normal(size=original.anchor.shape)
    d = rng.normal(size=original.auxiliary.shape)
    original.auxiliary = carrier.project(original.anchor) + d
    original.value = original.anchor + carrier.adjoint(d)
    coefficient = coordinates(original)
    gram = joint_matrix(carrier)
    original_energy = cp.sum(coefficient * (gram @ coefficient), axis=0)
    assert float(cp.max(cp.abs(original_energy - original.energy()))) < 1e-11
    records = []
    for duration, amplitude in ((0.0, 0.5), (0.2, 0.0), (0.2, 0.5)):
        operator = ReactionMoments(carrier, duration, amplitude, diagnostics=True)
        weighted = joint_matrix(carrier, operator)
        squared = joint_matrix(carrier, operator, 2)
        positive = float(cp.linalg.eigvalsh((weighted + weighted.T) / 2).min())
        bound = operator.decay_bound * gram - weighted
        coercivity = float(cp.linalg.eigvalsh((bound + bound.T) / 2).min())
        assert min(positive, coercivity) > -2e-10, (positive, coercivity)
        if amplitude == 0 or duration == 0:
            exact_weight = cp.exp(cp.asarray(-duration))
            uniform_error = float(cp.max(cp.abs(weighted - exact_weight * gram)))
            assert uniform_error < 1e-11, uniform_error
        else:
            refined = ReactionMoments(
                carrier, duration, amplitude, subdivisions=2, diagnostics=True
            )
            uniform_error = None
            quadrature_error = float(
                cp.max(cp.abs(joint_matrix(carrier, refined) - weighted))
            )
            assert quadrature_error < 1e-10, quadrature_error
        for shared in (False, True):
            state = copy(original)
            diagnostic = operator.apply(state, shared=shared, krylov_seed=krylov_seed)
            new = coordinates(state)
            target_energy = cp.sum(coefficient * (squared @ coefficient), axis=0)
            after = cp.sum(new * (gram @ new), axis=0)
            inner = cp.sum(coefficient * (weighted @ new), axis=0)
            distance = target_energy + after - 2 * inner
            pythagoras = float(cp.max(cp.abs(distance - (target_energy - after))))
            energy_oracle = float(
                cp.max(cp.abs(target_energy - diagnostic["target_energy"]))
            )
            observation = (
                carrier.adjoint(state.auxiliary)
                + state.anchor
                - carrier.adjoint(carrier.project(state.anchor))
            )
            observation_error = float(cp.max(cp.abs(observation - state.value)))
            excess = float(cp.max(diagnostic["coercivity_excess"]))
            assert max(pythagoras, energy_oracle, observation_error, excess) < 2e-9, (
                pythagoras,
                energy_oracle,
                observation_error,
                excess,
            )
            unchanged_error = None
            if amplitude == 0 or duration == 0:
                factor = cp.exp(cp.asarray(-duration))
                unchanged_error = max(
                    float(cp.max(cp.abs(state.value - factor * original.value))),
                    float(
                        cp.max(cp.abs(state.auxiliary - factor * original.auxiliary))
                    ),
                )
                assert unchanged_error < 1e-9, unchanged_error
            records.append(
                dict(
                    degree=degree,
                    krylov_seed=krylov_seed,
                    duration=duration,
                    amplitude=amplitude,
                    shared=shared,
                    minimum_weighted_eigenvalue=positive,
                    minimum_coercivity_eigenvalue=coercivity,
                    uniform_matrix_error=uniform_error,
                    quadrature_error=(
                        None if uniform_error is not None else quadrature_error
                    ),
                    pythagoras_error=pythagoras,
                    energy_oracle_error=energy_oracle,
                    observation_error=observation_error,
                    coercivity_excess=excess,
                    uniform_state_error=unchanged_error,
                )
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed", choices=("moment", "increment", "residual"), default="moment"
    )
    args = parser.parse_args()
    meta = provenance()
    results = []
    for degree in (1, 2, 4):
        records = validate(degree, args.seed)
        results.extend(records)
        print(json.dumps(records), flush=True)
        args.output.write_text(
            json.dumps(dict(results=results, provenance=meta), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
