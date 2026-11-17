"""Dense CUDA proof oracles for the orthogonal graph transport chart.

Distributed under the Boost Software License, Version 1.0.
The dense reference uses direct solves. The numerical realization uses only
fixed-degree matrix-free Chebyshev inverses, with spectrum bounded by [1,2].
"""

import argparse
from copy import copy
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from carrier import Carrier
from tensor_grid import TensorGrid
from tensor_cellular import TensorCellular
from graph_memory import GraphMemory, inverse_graph_mass
from sharpened_graph_memory import SharpenedGraphMemory
from reaction_moments_validation import joint_matrix
from campaign import provenance


def graph_matrix(coupling, sharpened=False):
    nv = coupling.space.mesh.cells * coupling.space.nb
    nw = coupling.n**2 * coupling.nc
    gram = joint_matrix(coupling)
    r = gram[nv:, :nv]
    if sharpened:
        r = 2 * r @ cp.linalg.solve(cp.eye(nv) + r.T @ r, cp.eye(nv))
    inverse = cp.linalg.solve(cp.eye(nv) + r.T @ r, cp.eye(nv))
    blocks = (
        cp.concatenate((2 * r @ inverse @ r.T - cp.eye(nw), 2 * r @ inverse), axis=1),
        cp.concatenate((2 * inverse @ r.T, 2 * inverse - cp.eye(nv)), axis=1),
    )
    return cp.concatenate(blocks), r, inverse


def coordinates(state):
    c = state.coupling
    mass = cp.broadcast_to(c.mass, (c.n**2, c.nc))
    w = (
        (state.auxiliary * cp.sqrt(mass)[:, None, :])
        .transpose(0, 2, 1)
        .reshape(-1, state.value.shape[1])
    )
    z = (
        (state.hidden_fe * cp.sqrt(c.space.mesh.areas)[:, None, None])
        .transpose(0, 2, 1)
        .reshape(-1, state.value.shape[1])
    )
    return cp.concatenate((w, z))


def validate(degree, sharpened=False):
    space = Space(Mesh(4, jitter=0.18, seed=817), degree)
    old = Carrier(space, n=4, builder="tiled")
    line = cp.linspace(0, 1, 5)
    edges = cp.stack(
        (
            line + 0.03 * cp.sin(2 * cp.pi * line) + 0.06,
            line - 0.02 * cp.sin(2 * cp.pi * line) - 0.03,
        )
    )
    new = Carrier(space, n=4, edges=edges, builder="tiled")
    grid = TensorGrid(new, new.edges)
    cross = lambda value: grid.remap(old, value)
    u0, r0, inverse0 = graph_matrix(old, sharpened)
    u1, r1, inverse1 = graph_matrix(new, sharpened)
    nw, nv = r0.shape
    wi = cp.ascontiguousarray(
        cp.eye(nw).reshape(old.n**2, old.nc, nw).transpose(0, 2, 1)
    )
    c = cross(wi).transpose(0, 2, 1).reshape(nw, nw)
    oldmass = cp.broadcast_to(old.mass, (old.n**2, old.nc)).ravel()
    newmass = cp.broadcast_to(new.mass, (new.n**2, new.nc)).ravel()
    c *= cp.sqrt(newmass[:, None] / oldmass[None, :])
    transport = cp.zeros((nw + nv, nw + nv))
    transport[:nw, :nw], transport[nw:, nw:] = c, cp.eye(nv)
    b = u1 @ transport @ u0
    norm = float(cp.linalg.svd(b, compute_uv=False)[0])
    observation_defect = float(cp.max(cp.abs(u1[:, nw:].T @ b - u0[:, nw:].T)))
    minimal_defect = float(cp.max(cp.abs(b @ u0[:, nw:] - u1[:, nw:])))
    orthogonality = max(
        float(cp.max(cp.abs(u.T @ u - cp.eye(nw + nv)))) for u in (u0, u1)
    )
    assert (
        norm < 1 + 1e-11
        and max(observation_defect, minimal_defect, orthogonality) < 1e-11
    )
    random = cp.random.RandomState(928)
    fields = random.normal(size=(space.mesh.cells, 4, space.nb))
    records = []
    create = SharpenedGraphMemory if sharpened else GraphMemory
    for polynomial_degree in (32, 40, 48) if sharpened else (12, 16, 20):
        state = create(old, fields, degree=polynomial_degree)
        weighted_fields = (
            (fields * cp.sqrt(space.mesh.areas)[:, None, None])
            .transpose(0, 2, 1)
            .reshape(nv, -1)
        )
        initialization_error = float(
            cp.linalg.norm(coordinates(state) - u0[:, nw:] @ weighted_fields)
            / cp.linalg.norm(weighted_fields)
        )
        # Random augmented states also test hidden directions not reached by initialization.
        state.auxiliary = random.normal(size=state.auxiliary.shape)
        state.hidden_fe = random.normal(size=state.hidden_fe.shape)
        state.value = state.observe()
        initial = coordinates(state)
        expected = b @ initial
        same = copy(state)
        same.transfer(old, lambda x: x)
        identity_error = float(
            cp.linalg.norm(coordinates(same) - initial) / cp.linalg.norm(initial)
        )
        old_value = state.value.copy()
        diagnostic = state.transfer(new, cross)
        actual = coordinates(state)
        error = float(cp.linalg.norm(actual - expected) / cp.linalg.norm(expected))
        observed_error = float(cp.max(cp.abs(state.observe() - old_value)))
        exact_preservation = bool(cp.array_equal(state.value, old_value))
        energy_identity = float(
            cp.max(
                cp.abs(diagnostic["energy_loss"] - diagnostic["hidden_transfer_loss"])
            )
        )
        rhs = fields
        if sharpened:
            solver = create(old, fields, degree=polynomial_degree)
            solved = solver.solve(rhs)
            action = solver.gram(solved)
            residual = rhs - solved - 6 * action - solver.gram(action)
        else:
            solved = inverse_graph_mass(old, rhs, polynomial_degree)
            residual = rhs - solved - old.adjoint(old.project(solved))
        inverse_error = float(
            cp.sqrt(cp.max(space.energy(residual) / space.energy(rhs)))
        )
        physical = copy(state)
        decay = 0.4
        factors = random.uniform(0.3, 0.95, size=(space.mesh.cells, 1, space.nb))
        delta = (factors - decay) * physical.value
        weighted_delta = (
            (delta * cp.sqrt(space.mesh.areas)[:, None, None])
            .transpose(0, 2, 1)
            .reshape(nv, -1)
        )
        expected_physics = decay * actual + u1[:, nw:] @ weighted_delta
        energy_before = physical.energy()
        physical.apply_physics(lambda value: factors * value, hidden_decay=decay)
        physics_error = float(
            cp.linalg.norm(coordinates(physical) - expected_physics)
            / cp.linalg.norm(expected_physics)
        )
        physics_observation_error = float(
            cp.max(cp.abs(physical.observe() - factors * old_value))
        )
        physics_energy_ratio = float(cp.max(physical.energy() / energy_before))
        tolerance = {12: 1e-7, 16: 1e-10, 20: 1e-11, 32: 1e-8, 40: 1e-10, 48: 1e-11}[
            polynomial_degree
        ]
        assert (
            max(
                error,
                observed_error,
                identity_error,
                energy_identity,
                inverse_error,
                initialization_error,
                physics_error,
                physics_observation_error,
            )
            < tolerance
        )
        assert exact_preservation
        assert physics_energy_ratio <= 1 + tolerance
        records.append(
            dict(
                degree=degree,
                sharpened=sharpened,
                inverse_degree=polynomial_degree,
                exact_transition_norm=norm,
                dense_orthogonality=orthogonality,
                dense_observation_defect=observation_defect,
                dense_minimal_state_defect=minimal_defect,
                state_oracle_relative_error=error,
                observed_error=observed_error,
                coefficient_preservation=exact_preservation,
                identity_relative_error=identity_error,
                energy_identity_error=energy_identity,
                inverse_relative_residual=inverse_error,
                initialization_relative_error=initialization_error,
                physics_relative_error=physics_error,
                physics_observation_error=physics_observation_error,
                physics_energy_ratio=physics_energy_ratio,
            )
        )
    state = create(old, fields)
    flow = TensorCellular(old, 0.01, grading=0.0)
    energy0, mass0 = state.energy(), space.mass(state.value)
    for _ in range(8):
        state.advect(flow.apply_carrier)
    ratio = float(cp.max(state.energy() / energy0))
    mass_error = float(cp.max(cp.abs(space.mass(state.value) - mass0)))
    observation_error = float(cp.max(cp.abs(state.value - state.observe())))
    assert ratio <= 1 + 1e-10 and max(mass_error, observation_error) < 1e-10
    return records, dict(
        degree=degree,
        energy_ratio=ratio,
        mass_error=mass_error,
        observation_error=observation_error,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sharpened", action="store_true")
    args = parser.parse_args()
    result = dict(provenance=provenance(), controls=[], advection=[])
    for degree in (1, 2, 4):
        controls, flow = validate(degree, args.sharpened)
        result["controls"].extend(controls)
        result["advection"].append(flow)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(dict(degree=degree, controls=controls, advection=flow)),
            flush=True,
        )


if __name__ == "__main__":
    main()
