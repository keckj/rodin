"""GPU tests of the zero-motion defect and a nonlinear warm-start transfer.

Distributed under the Boost Software License, Version 1.0.
Energy stability, FE preservation and no-op consistency are separate tests.
This does not claim linearity or a pairwise contraction for the warm transfer.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from gpu import cp, Space, Mesh
from carrier import Carrier
from tensor_grid import TensorGrid
from memory_transport import MemoryTransport
from campaign import provenance


def limiter_counterexample():
    """A three-dimensional ambient oracle separates energy from Lipschitz stability."""
    space = SimpleNamespace(energy=lambda v: cp.sum(v**2, axis=(0, 2)))

    def coupling(angle):
        c, s = cp.cos(cp.asarray(angle)), cp.sin(cp.asarray(angle))
        r = cp.stack((c, cp.asarray(0.0))).reshape(1, 1, 2)
        embedding = cp.stack(
            (cp.stack((c, s, cp.asarray(0.0))), cp.asarray([0.0, 0.0, 1.0])), axis=1
        )
        return SimpleNamespace(
            space=space,
            project=lambda v: v * r,
            adjoint=lambda w: cp.sum(w * r, axis=2, keepdims=True),
            energy=lambda w: cp.sum(w**2, axis=(0, 2)),
            embedding=embedding,
        )

    theta0, theta1 = 0.05, 0.5
    old, new = coupling(theta0), coupling(theta1)
    epsilon = 0.001
    coordinates = cp.asarray([[epsilon, -epsilon, 0.0], [1.0, 1.0, 2.0]])
    state = MemoryTransport(old, cp.zeros((1, 3, 1)))
    state.anchor = (-coordinates[0] / cp.tan(cp.asarray(theta0))).reshape(1, 3, 1)
    state.auxiliary = cp.stack(
        (cp.sin(cp.asarray(theta0)) * coordinates[0], coordinates[1]), axis=1
    )[None, ...]
    original = cp.concatenate((cp.zeros((1, 3)), coordinates), axis=0)
    cross = (
        lambda w: w
        * cp.stack((cp.cos(cp.asarray(theta1 - theta0)), cp.asarray(1.0)))[
            None, None, :
        ]
    )
    diagnostic = state.project_memory_warm(new, cross, iterations=8)
    q = new.embedding @ (state.auxiliary - new.project(state.anchor))[0].T
    q[0] += state.anchor[0, :, 0]
    energy_excess = float(cp.max(cp.sum(q**2 - original**2, axis=0)))
    pair_gain = float(
        cp.linalg.norm(q[:, 0] - q[:, 1])
        / cp.linalg.norm(original[:, 0] - original[:, 1])
    )
    additivity_error = float(cp.linalg.norm(q[:, 0] + q[:, 1] - q[:, 2]))
    assert energy_excess < 1e-12 and float(cp.min(diagnostic["limiter"])) < 1
    assert pair_gain > 1.1 and additivity_error > 1e-8
    return dict(
        energy_excess=energy_excess,
        pair_distance_gain=pair_gain,
        additivity_error=additivity_error,
        limiter=diagnostic["limiter"].tolist(),
    )


def validate(p, methods=("cold", "warm")):
    space = Space(Mesh(8, jitter=0.18, seed=817), p)
    carrier = Carrier(space, n=8)
    initial = cp.random.RandomState(813).normal(size=(space.mesh.cells, 4, space.nb))
    original = MemoryTransport(carrier, initial)
    original.advect(carrier.transport(0.07, "shear").apply_carrier)
    original.apply_physics(lambda v: 0.9 * v)
    original.advect(carrier.transport(-0.04, "translation").apply_carrier)
    before = original.energy()
    records = []
    for motion in (0.0, 1e-6, 1e-4, 1e-2, 0.1):
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
        for method in methods:
            state = MemoryTransport(carrier, original.value)
            state.anchor = original.anchor.copy()
            state.auxiliary = original.auxiliary.copy()
            if method == "cold":
                diagnostic = {}
                state.project_memory(target, cross, 8, diagnostics=diagnostic)
                scale = None
            elif method == "warm":
                diagnostic = state.project_memory_warm(target, cross, 8)
                scale = diagnostic["limiter"].tolist()
            elif method.startswith("galerkin"):
                diagnostic = state.project_memory_galerkin(
                    target, cross, int(method.removeprefix("galerkin"))
                )
                scale = None
            else:
                raise ValueError("Unknown validation method")
            energy_excess = float(cp.max(state.energy() - before))
            observation = (
                target.adjoint(state.auxiliary)
                + state.anchor
                - target.adjoint(target.project(state.anchor))
            )
            observation_error = float(cp.max(cp.abs(observation - original.value)))
            assert energy_excess < 1e-10 and observation_error < 1e-10
            # Form the old/new ambient cross product independently, without
            # the h,x formula used by the production diagnostic.
            d = original.auxiliary - carrier.project(original.anchor)
            e = state.auxiliary - target.project(state.anchor)
            inner = (
                (
                    space.energy(original.anchor + state.anchor)
                    - space.energy(original.anchor - state.anchor)
                )
                / 4
                + (
                    carrier.energy(d + carrier.project(state.anchor))
                    - carrier.energy(d - carrier.project(state.anchor))
                )
                / 4
                + (
                    target.energy(e + target.project(original.anchor))
                    - target.energy(e - target.project(original.anchor))
                )
                / 4
                + (target.energy(e + cross(d)) - target.energy(e - cross(d))) / 4
            )
            distance = before + state.energy() - 2 * inner
            oracle_error = float(
                cp.max(cp.abs(distance - diagnostic["ambient_change_squared"]))
            )
            assert oracle_error < 1e-10, oracle_error
            if method.startswith("galerkin"):
                assert (
                    float(cp.max(cp.abs(distance - diagnostic["energy_loss"]))) < 2e-9
                )
            if motion == 0 and method != "cold":
                assert float(cp.max(cp.abs(distance))) < 1e-10
                assert (
                    float(cp.max(cp.abs(state.auxiliary - original.auxiliary))) < 1e-10
                )
            records.append(
                dict(
                    motion=motion,
                    method=method,
                    energy_excess=energy_excess,
                    observation_error=observation_error,
                    oracle_error=oracle_error,
                    ambient_change_squared=distance.tolist(),
                    energy_loss=diagnostic["energy_loss"].tolist(),
                    limiter=scale,
                )
            )
    for motion in (0.0, 1e-6, 1e-4, 1e-2, 0.1):
        augmented = sorted(
            [
                z
                for z in records
                if z["motion"] == motion and z["method"].startswith("galerkin")
            ],
            key=lambda z: int(z["method"].removeprefix("galerkin")),
        )
        for old, new in zip(augmented, augmented[1:]):
            # More directions span a larger ambient projection space.
            assert (
                max(b - a for a, b in zip(old["energy_loss"], new["energy_loss"]))
                < 2e-9
            )
    return dict(degree=p, initial_energy=before.tolist(), records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [validate(p) for p in (1, 2, 4)]
    args.output.write_text(
        json.dumps(
            dict(
                results=records,
                limiter_counterexample=limiter_counterexample(),
                provenance=provenance(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
