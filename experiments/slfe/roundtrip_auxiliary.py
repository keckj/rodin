"""Isolate auxiliary transport/remapping from FE memory in long return tests.

Distributed under the Boost Software License, Version 1.0.
This endpoint-only FE baseline does not support intermediate FE physics.
The reversible reference diagnoses dissipation, not every transport phase error.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Space
from graded_study import GradedMesh
from moving_study import profile_family
from carrier import Carrier
from tensor_grid import TensorGrid
from tensor_cellular import TensorCellular
from energy_mesh import from_fe, from_tensor
from campaign import provenance


def evolve(
    naux=128, steps=128, check_every=4, uniform=False, half_duration=1.0, nfe=256
):
    space = Space(GradedMesh(nfe, 0.95), 4)
    profile = profile_family(0.01, 4, spread=0.15)
    initial = space.project(profile, degree=20)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    if uniform:
        edges = cp.stack((cp.linspace(0, 1, naux + 1),) * 2)
        oriented = edges.copy()
    else:
        edges, oriented = from_fe(space, initial, naux)
    carrier = Carrier(space, n=naux, edges=edges)
    grid = TensorGrid(carrier, edges)
    value = carrier.project(initial)
    initial_defect = space.energy(initial) - grid.energy(value)
    mass0, energy0 = space.mass(initial), space.energy(initial)
    del carrier
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    history = []
    start = time.perf_counter()
    for leg, direction in enumerate((1, -1)):
        flow = TensorCellular(
            grid, direction * half_duration / steps, oriented_edges=oriented
        )
        for first in range(0, steps, check_every):
            count = min(check_every, steps - first)
            for _ in range(count):
                value = flow.apply_carrier(value)
            changed = False
            loss = None
            if not uniform and not (leg == 1 and first + count == steps):
                next_edges, next_oriented = from_tensor(grid, value)
                displacement = max(
                    float(cp.max(cp.abs(next_edges - grid.edges))),
                    float(cp.max(cp.abs(next_oriented - oriented))),
                )
                changed = displacement > 0.01
                if changed:
                    target = TensorGrid(grid, next_edges)
                    projected = target.remap(grid, value)
                    loss = (grid.energy(value) - target.energy(projected)).tolist()
                    grid, value, oriented = target, projected, next_oriented
                    flow = TensorCellular(
                        grid, direction * half_duration / steps, oriented_edges=oriented
                    )
            mass = cp.sum(value * grid.mass[:, None, :], axis=(0, 2))
            mass_error = float(cp.max(cp.abs(mass - mass0)))
            energy_ratio = float(cp.max(grid.energy(value) / energy0))
            assert mass_error < 3e-8 and energy_ratio <= 1 + 3e-8
            history.append(
                dict(
                    leg=leg,
                    step=first + count,
                    changed=changed,
                    remap_loss=loss,
                    mass_error=mass_error,
                    energy_ratio=energy_ratio,
                )
            )
    carrier = Carrier(space, n=naux, edges=grid.edges)
    result = carrier.adjoint(value)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    errors = space.errors(result, profile, 0, degree=20, subdivisions=2)
    floor = space.errors(initial, profile, 0, degree=20, subdivisions=2)["l2"]
    return dict(
        nfe=nfe,
        naux=naux,
        fields=4,
        steps_per_leg=steps,
        check_every=check_every,
        uniform=uniform,
        half_duration=half_duration,
        changes=sum(h["changed"] for h in history),
        initial_projection_loss=initial_defect.tolist(),
        setup_seconds=setup,
        evolution_and_final_projection_seconds=elapsed,
        best_fe_l2=floor,
        history=history,
        **errors
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        dict(naux=n, steps=s, check_every=c)
        for n in (128, 256)
        for s, c in ((128, 4), (512, 4), (512, 16))
    ]
    cases += [dict(naux=n, steps=512, uniform=True) for n in (128, 256)]
    meta = provenance()
    with args.output.open("w") as stream:
        for parameters in cases:
            print(json.dumps(parameters), flush=True)
            result = evolve(**parameters)
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
