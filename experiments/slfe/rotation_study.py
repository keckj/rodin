"""GPU tests of mass-preserving orthogonal local DG splitting.

Distributed under the Boost Software License, Version 1.0.
"""

import argparse
import json
from pathlib import Path
import time
import traceback

from gpu import cp, Space, gaussian
from graded_study import GradedMesh
from rotation_dg import RotationDG
from gpu_graph import EvolutionGraph
from day_study import exact_field
from campaign import provenance


def validate(p=2, grading=0.0):
    from cupyx.scipy.linalg import expm

    space = Space(GradedMesh(4, grading), p)
    flow = RotationDG(space)
    nv = space.mesh.cells * space.nb
    eye = cp.ascontiguousarray(
        cp.eye(nv).reshape(space.mesh.cells, space.nb, nv).transpose(0, 2, 1)
    )
    mass = cp.repeat(space.mesh.areas, space.nb)
    root = cp.sqrt(mass)

    def dense(value):
        return value.transpose(0, 2, 1).reshape(nv, nv)

    A = dense(flow.dg.rhs(eye))
    normal = root[:, None] * A / root[None, :]
    c = cp.zeros(nv)
    c[:: space.nb] = cp.sqrt(space.mesh.areas)
    P = cp.eye(nv) - cp.outer(c, c)
    target = P @ normal @ P
    invariants = []
    for dt in (0.01, 1.0, -1.0):
        op = flow.prepare(dt, 4)
        matrix = dense(op.step(eye)) * root[:, None] / root[None, :]
        constant = float(cp.max(cp.abs(matrix @ c - c)))
        orthogonal = float(cp.max(cp.abs(matrix.T @ matrix - cp.eye(nv))))
        opposite = dense(flow.prepare(-dt, 4).step(eye)) * root[:, None] / root[None, :]
        reverse = float(cp.max(cp.abs(opposite @ matrix - cp.eye(nv))))
        assert max(constant, orthogonal, reverse) < 3e-11, (
            constant,
            orthogonal,
            reverse,
        )
        invariants.append(
            dict(
                dt=dt,
                constant_error=constant,
                orthogonal_error=orthogonal,
                reversal_error=reverse,
                minimum_overlap=op.minimum_overlap,
            )
        )
    duration = 0.03
    exact = expm(duration * target)
    temporal = []
    for order in (2, 4):
        errors = []
        for steps in (8, 16, 32, 64):
            op = flow.prepare(duration / steps, order)
            one = dense(op.step(eye)) * root[:, None] / root[None, :]
            full = cp.linalg.matrix_power(one, steps)
            errors.append(float(cp.linalg.norm(full - exact)))
        ratio = errors[-2] / errors[-1]
        assert 3.8 < ratio < 4.2 if order == 2 else 14.5 < ratio < 17.5, (
            order,
            errors,
            ratio,
        )
        temporal.append(dict(order=order, errors=errors, ratio=ratio))
    value = space.project(gaussian)
    op = flow.prepare(0.01, 4)
    graph = EvolutionGraph(op.step, value, 0.01)
    graph.launch(0.01)
    difference = float(cp.max(cp.abs(graph.result() - op.step(value))))
    assert difference < 3e-12, difference
    return dict(
        p=p,
        grading=grading,
        groups=len(flow.groups),
        coloring_rounds=flow.coloring_rounds,
        generator_projection_difference=float(cp.max(cp.abs(target - normal))),
        invariants=invariants,
        temporal=temporal,
        graph_difference=difference,
    )


def evolve(n=16, p=4, grading=0.95, steps=32, order=4, end=0.5):
    space = Space(GradedMesh(n, grading), p)
    initial = space.project(gaussian)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    flow = RotationDG(space)
    duration = end / steps
    op = flow.prepare(duration, order)
    graph = EvolutionGraph(op.step, initial, duration)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(steps):
        graph.launch(duration)
    value = graph.result()
    elapsed = time.perf_counter() - start
    errors = space.errors(value, lambda points, t: exact_field(points, t, False), end)
    mass = float(cp.max(cp.abs(space.mass(value) - space.mass(initial))))
    energy = float(cp.max(cp.abs(space.energy(value) / space.energy(initial) - 1)))
    assert max(mass, energy) < 3e-9, (mass, energy)
    return dict(
        n=n,
        p=p,
        grading=grading,
        steps=steps,
        order=order,
        end=end,
        groups=len(flow.groups),
        stages=len(op.stages),
        static_setup_seconds=setup,
        evolution_seconds=elapsed,
        mass_error=mass,
        energy_error=energy,
        minimum_overlap=op.minimum_overlap,
        **errors
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=("validation", "accuracy", "refinement"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = (
        [dict(p=p, grading=g) for p in (1, 2, 3) for g in (0.0, 0.8)]
        if args.suite == "validation"
        else [
            dict(n=n, grading=g, steps=s)
            for n in (8, 16, 32)
            for g in (0.0, 0.95)
            for s in (16, 32, 64, 128)
        ]
    )
    if args.suite == "refinement":
        cases = [
            dict(n=8, grading=g, steps=s)
            for g in (0.0, 0.95)
            for s in (256, 512, 1024, 2048)
        ]
    meta = provenance()
    with (args.output / ("rotation-" + args.suite + ".jsonl")).open("w") as f:
        for case in cases:
            try:
                r = dict(
                    status="ok",
                    result=(
                        validate(**case)
                        if args.suite == "validation"
                        else evolve(**case)
                    ),
                )
            except Exception:
                r = dict(status="failed", error=traceback.format_exc())
            r.update(parameters=case, provenance=meta)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
