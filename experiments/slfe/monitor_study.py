"""GPU ablations of stationary mesh drift and adaptation-check frequency.

Distributed under the Boost Software License, Version 1.0.
The stationary experiment contains no transport and cannot confuse monitor
feedback with time integration. Old policies remain explicit controls.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Space
from graded_study import GradedMesh
from carrier import Carrier
from tensor_grid import TensorGrid
from moving_study import profile_family
from energy_mesh import from_fe, from_tensor, tensor_samples
from roundtrip_study import evolve
from campaign import provenance


def stationary(naux=64, sample_order=5, energy_floor=0.0, iterations=24):
    space = Space(GradedMesh(128, 0.95), 4)
    initial = space.project(profile_family(0.01, 4, spread=0.15), degree=20)
    edges, oriented = from_fe(space, initial, naux, energy_floor=energy_floor)
    carrier = Carrier(space, n=naux, edges=edges)
    grid = TensorGrid(carrier, edges)
    values = carrier.project(initial)
    initial_values = values.copy()
    energy0 = grid.energy(values)
    mass0 = cp.sum(grid.mass[:, None, :] * values, axis=(0, 2))
    _, weights = tensor_samples(grid, values, sample_order)
    quadrature_error = abs(float(cp.sum(weights) - cp.sum(energy0)))
    assert quadrature_error < 1e-11, quadrature_error
    history = []
    for step in range(iterations):
        next_edges, next_oriented = from_tensor(
            grid, values, sample_order=sample_order, energy_floor=energy_floor
        )
        displacement = max(
            float(cp.max(cp.abs(next_edges - grid.edges))),
            float(cp.max(cp.abs(next_oriented - oriented))),
        )
        target = TensorGrid(carrier, next_edges)
        previous = grid.energy(values)
        values = target.remap(grid, values)
        grid, oriented = target, next_oriented
        ratio = float(cp.max(grid.energy(values) / previous))
        mass_error = float(
            cp.max(cp.abs(cp.sum(grid.mass[:, None, :] * values, axis=(0, 2)) - mass0))
        )
        assert ratio < 1 + 1e-10 and mass_error < 1e-10, (ratio, mass_error)
        history.append(
            dict(
                iteration=step + 1,
                displacement=displacement,
                maximum_energy_ratio=ratio,
                mass_error=mass_error,
                lost_energy=(energy0 - grid.energy(values)).tolist(),
            )
        )
    returned = TensorGrid(carrier, carrier.edges).remap(grid, values)
    return dict(
        kind="stationary",
        naux=naux,
        sample_order=sample_order,
        energy_floor=energy_floor,
        quadrature_error=quadrature_error,
        return_l2=cp.sqrt(carrier.energy(returned - initial_values)).tolist(),
        history=history,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("stationary", "return"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.suite == "stationary":
        cases = [
            dict(naux=n, sample_order=q, energy_floor=f)
            for n in (32, 64, 128)
            for q in (5, 12)
            for f in (0.0, 1e-6)
        ]
    else:
        cases = [
            dict(steps=steps, check_every=interval, sample_order=q, energy_floor=f)
            for steps, interval in ((128, 4), (512, 4), (512, 16))
            for q, f in ((5, 0.0), (12, 0.0), (5, 1e-6), (12, 1e-6))
        ]
    meta = provenance()
    with args.output.open("w") as stream:
        for case in cases:
            print(json.dumps(case), flush=True)
            result = (
                stationary(**case) if args.suite == "stationary" else evolve(**case)
            )
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
