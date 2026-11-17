"""GPU cross-field Gram and coordinate-covariance tests for shared memory spaces.

Distributed under the Boost Software License, Version 1.0.
The independent ambient product retains all cross-field entries, rather than
only separate energies. Orthogonal field mixing is tested at fixed geometry.
"""

import argparse
from copy import copy
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from carrier import Carrier
from tensor_grid import TensorGrid
from memory_transport import MemoryTransport
from campaign import provenance


def product(a, b, mass):
    return cp.einsum("cfi,ci,cgi->fg", a, mass, b)


def energy_matrix(state):
    coupling = state.coupling
    vmass = cp.broadcast_to(coupling.space.mesh.areas[:, None], state.anchor.shape[::2])
    wmass = cp.broadcast_to(coupling.mass, state.auxiliary.shape[::2])
    projected = coupling.project(state.anchor)
    return (
        product(state.auxiliary, state.auxiliary, wmass)
        + product(state.anchor, state.anchor, vmass)
        - product(projected, projected, wmass)
    )


def ambient_product(old, new, cross):
    old_c, new_c = old.coupling, new.coupling
    vmass = cp.broadcast_to(old_c.space.mesh.areas[:, None], old.anchor.shape[::2])
    old_mass = cp.broadcast_to(old_c.mass, old.auxiliary.shape[::2])
    new_mass = cp.broadcast_to(new_c.mass, new.auxiliary.shape[::2])
    d = old.auxiliary - old_c.project(old.anchor)
    e = new.auxiliary - new_c.project(new.anchor)
    return (
        product(old.anchor, new.anchor, vmass)
        + product(d, old_c.project(new.anchor), old_mass)
        + product(new_c.project(old.anchor), e, new_mass)
        + product(cross(d), e, new_mass)
    )


def rotate(state, matrix):
    result = copy(state)
    for name in ("value", "anchor", "auxiliary"):
        setattr(
            result,
            name,
            cp.ascontiguousarray(
                cp.einsum("cfb,fg->cgb", getattr(state, name), matrix)
            ),
        )
    return result


def validate(degree, gram_backend="einsum"):
    space = Space(Mesh(8, jitter=0.18, seed=817), degree)
    carrier = Carrier(space, n=8)
    initial = cp.random.RandomState(813).normal(size=(space.mesh.cells, 4, space.nb))
    original = MemoryTransport(carrier, initial)
    original.advect(carrier.transport(0.07, "shear").apply_carrier)
    original.apply_physics(lambda v: 0.9 * v)
    original.advect(carrier.transport(-0.04, "translation").apply_carrier)
    before = energy_matrix(original)
    rotation, _ = cp.linalg.qr(cp.random.RandomState(41).normal(size=(4, 4)))
    records = []
    for motion in (0.0, 1e-4, 0.01, 0.1):
        coordinate = cp.linspace(0, 1, carrier.n + 1)
        edges = cp.stack(
            (
                coordinate + motion * cp.sin(2 * cp.pi * coordinate) / (2 * cp.pi),
                coordinate,
            )
        )
        target = Carrier(space, n=8, edges=edges)
        grid = TensorGrid(target, target.edges)
        cross = lambda w: grid.remap(carrier, w)
        previous = None
        for iterations in (0, 1, 2, 4):
            state = copy(original)
            diagnostic = state.project_memory_galerkin(
                target, cross, iterations, shared=True, gram_backend=gram_backend
            )
            after = energy_matrix(state)
            inner = ambient_product(original, state, cross)
            distance = before + after - inner - inner.T
            loss = before - after
            pythagoras = float(cp.max(cp.abs(loss - distance)))
            minimum_loss = float(cp.linalg.eigvalsh((loss + loss.T) / 2).min())
            observation = (
                target.adjoint(state.auxiliary)
                + state.anchor
                - target.adjoint(target.project(state.anchor))
            )
            observation_error = float(cp.max(cp.abs(observation - original.value)))
            assert (
                pythagoras < 2e-9 and minimum_loss > -2e-9 and observation_error < 1e-10
            )
            enriched_gain = None
            if previous is not None:
                gain = after - previous
                enriched_gain = float(cp.linalg.eigvalsh((gain + gain.T) / 2).min())
                assert enriched_gain > -2e-9
            previous = after
            mixed = rotate(original, rotation)
            mixed.project_memory_galerkin(
                target, cross, iterations, shared=True, gram_backend=gram_backend
            )
            expected = rotate(state, rotation)
            covariance = max(
                float(cp.max(cp.abs(getattr(mixed, key) - getattr(expected, key))))
                for key in ("anchor", "auxiliary")
            )
            assert covariance < 2e-8, dict(
                covariance=covariance,
                degree=degree,
                motion=motion,
                iterations=iterations,
                rank=int(diagnostic["retained_rank"]),
            )
            independent = copy(original)
            independent.project_memory_galerkin(target, cross, iterations)
            mixed_independent = rotate(original, rotation)
            mixed_independent.project_memory_galerkin(target, cross, iterations)
            expected_independent = rotate(independent, rotation)
            individual_covariance = max(
                float(
                    cp.max(
                        cp.abs(
                            getattr(mixed_independent, key)
                            - getattr(expected_independent, key)
                        )
                    )
                )
                for key in ("anchor", "auxiliary")
            )
            # The shared space contains every individual-field space.
            benefit = independent.energy() - state.energy()
            assert float(cp.max(benefit)) < 2e-9
            if motion == 0:
                assert float(cp.max(cp.abs(distance))) < 2e-9
            records.append(
                dict(
                    motion=motion,
                    iterations=iterations,
                    rank=int(diagnostic["retained_rank"]),
                    pythagoras_error=pythagoras,
                    minimum_gram_loss_eigenvalue=minimum_loss,
                    minimum_enrichment_gain_eigenvalue=enriched_gain,
                    observation_error=observation_error,
                    covariance_error=covariance,
                    individual_covariance_error=individual_covariance,
                    retained_energy_benefit=(-benefit).tolist(),
                )
            )
    return dict(degree=degree, records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gram-backend", choices=("einsum", "reduction"), default="einsum"
    )
    args = parser.parse_args()
    meta = provenance()
    results = [validate(p, args.gram_backend) for p in (1, 2, 4)]
    args.output.write_text(
        json.dumps(dict(results=results, provenance=meta), indent=2) + "\n"
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
