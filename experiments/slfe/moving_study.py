"""Pure-advection baseline with moving tensor grids and endpoint FE projections.

Distributed under the Boost Software License, Version 1.0.
This baseline discards the initial FE projection defect and does not implement
exact intermediate FE physics. Its initial defect and remapping loss are logged.
"""

import argparse
import json
import math
from pathlib import Path
import time

from gpu import cp, Space
from graded_study import GradedMesh
from carrier import Carrier
from tensor_grid import TensorGrid
from tensor_cellular import TensorCellular
from adaptive_study import partitions, fe_center, tensor_center
from day_study import reference_feet
from campaign import provenance


def profile_family(width, fields, spread=0.008):
    random = cp.random.RandomState(752)
    centers = cp.asarray([0.35, 0.45])[None, :] + (
        cp.zeros((1, 2))
        if fields == 1
        else random.uniform(-spread, spread, (fields, 2))
    )
    widths = (
        cp.full(fields, width)
        if fields == 1
        else width * random.uniform(0.8, 1.2, fields)
    )

    def profile(points, t):
        delta = cp.sin(cp.pi * (points[..., None, :] - centers))
        return cp.exp(-cp.sum(delta * delta, axis=-1) / (2 * (cp.pi * widths) ** 2)) / (
            math.sqrt(math.pi) * widths
        )

    return profile


def evolve(
    nfe=128,
    naux=32,
    width=0.01,
    end=0.25,
    steps=128,
    check_every=4,
    threshold=0.03,
    grading=0.95,
    adaptive=True,
    fields=1,
    spread=0.008,
    mesh_policy="centroid",
    with_fields=False,
):
    space = Space(GradedMesh(nfe, 0.95), 4)
    profile = profile_family(width, fields, spread)
    exact = lambda points, t: profile(reference_feet(points, -end, 512), 0)
    initial = space.project(profile, degree=20)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    center = fe_center(space, initial)
    oriented = None
    if mesh_policy == "energy":
        from energy_mesh import from_fe, from_tensor

        edges, oriented = from_fe(space, initial, naux)
    elif mesh_policy == "centroid":
        edges = partitions(naux, grading, center)
    else:
        raise ValueError("Expected centroid or energy mesh policy")
    initial_carrier = Carrier(space, n=naux, edges=edges)
    grid = TensorGrid(initial_carrier, initial_carrier.edges)
    flow = TensorCellular(
        grid, end / steps, grading, center=center, oriented_edges=oriented
    )
    w = initial_carrier.project(initial)
    initial_energy = space.energy(initial)
    initial_defect = initial_energy - grid.energy(w)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    history = []
    rebuild = 0.0
    start = time.perf_counter()
    for first in range(0, steps, check_every):
        count = min(check_every, steps - first)
        for _ in range(count):
            w = flow.apply_carrier(w)
        detected, concentration = tensor_center(grid, w)
        displacement = (detected - center + 0.5) % 1 - 0.5
        distance = float(cp.linalg.norm(displacement))
        change = adaptive and first + count < steps and distance > threshold
        next_edges, next_oriented = None, None
        if mesh_policy == "energy":
            next_edges, next_oriented = from_tensor(grid, w)
            movement = max(
                float(cp.max(cp.abs(next_edges - grid.edges))),
                float(cp.max(cp.abs(next_oriented - oriented))),
            )
            # Fixed endpoints make edge displacement meaningful. One third
            # of the centroid threshold is an exploratory mesh-motion trigger.
            change = adaptive and first + count < steps and movement > threshold / 3
        loss = None
        if change:
            cp.cuda.Stream.null.synchronize()
            before = time.perf_counter()
            next_grid = TensorGrid(
                grid,
                (
                    next_edges
                    if next_edges is not None
                    else partitions(naux, grading, detected)
                ),
            )
            next_w = next_grid.remap(grid, w)
            loss = grid.energy(w) - next_grid.energy(next_w)
            assert float(cp.min(loss)) > -2e-10
            grid, w, center = next_grid, next_w, detected.copy()
            oriented = next_oriented
            flow = TensorCellular(
                grid, end / steps, grading, center=center, oriented_edges=oriented
            )
            cp.cuda.Stream.null.synchronize()
            rebuild += time.perf_counter() - before
        energy_ratio = float(cp.max(grid.energy(w) / initial_energy))
        mass_error = float(
            cp.max(
                cp.abs(
                    cp.sum(grid.mass[:, None, :] * w, axis=(0, 2)) - space.mass(initial)
                )
            )
        )
        assert mass_error < 2e-9 and energy_ratio < 1 + 2e-9, (mass_error, energy_ratio)
        history.append(
            dict(
                step=first + count,
                center=center.tolist(),
                detected=detected.tolist(),
                concentration=concentration.tolist(),
                displacement=distance,
                restarted=change,
                remap_loss=None if loss is None else loss.tolist(),
                mass_error=mass_error,
                energy_ratio=energy_ratio,
            )
        )
    cp.cuda.Stream.null.synchronize()
    evolution = time.perf_counter() - start
    start = time.perf_counter()
    if adaptive and any(row["restarted"] for row in history):
        final_carrier = Carrier(space, n=naux, edges=grid.edges)
    else:
        final_carrier = initial_carrier
    result = final_carrier.adjoint(w)
    cp.cuda.Stream.null.synchronize()
    finalization = time.perf_counter() - start
    errors = space.errors(result, exact, 0, degree=20, subdivisions=2)
    best = space.errors(
        space.project(exact, degree=20), exact, 0, degree=20, subdivisions=2
    )["l2"]
    record = dict(
        nfe=nfe,
        naux=naux,
        width=width,
        end=end,
        steps=steps,
        fields=fields,
        spread=spread,
        mesh_policy=mesh_policy,
        threshold=threshold,
        check_every=check_every,
        adaptive=adaptive,
        grading=grading,
        setup_seconds=setup,
        evolution_seconds=evolution,
        finalization_seconds=finalization,
        total_seconds=setup + evolution + finalization,
        rebuild_seconds=rebuild,
        restarts=sum(r["restarted"] for r in history),
        initial_defect_squared=initial_defect.tolist(),
        best_fe_l2=best,
        history=history,
        **errors
    )
    return (space, initial, result, exact, record) if with_fields else record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=("smoke", "resolution", "fields", "separated"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.suite == "smoke":
        cases = [dict(naux=n, adaptive=a) for n in (32, 64) for a in (False, True)]
    elif args.suite == "resolution":
        cases = [
            dict(naux=n, end=end, steps=steps, adaptive=a)
            for n in (16, 32, 64)
            for end in (0.1, 0.25)
            for steps in (16, 64, 256)
            for a in (False, True)
        ]
    elif args.suite == "separated":
        cases = [
            dict(naux=n, fields=4, width=0.01, spread=s, steps=128, mesh_policy=m)
            for s in (0.008, 0.05, 0.15)
            for n in (32, 64, 128)
            for m in ("centroid", "energy")
        ]
    else:
        cases = [
            dict(naux=n, fields=f, steps=steps)
            for n in (32, 64)
            for f in (1, 4, 16)
            for steps in (16, 64, 256)
        ]
    meta = provenance()
    with args.output.open("w") as stream:
        for parameters in cases:
            print(json.dumps(parameters), flush=True)
            result = evolve(**parameters)
            stream.write(
                json.dumps(dict(parameters=parameters, result=result, provenance=meta))
                + "\n"
            )
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
