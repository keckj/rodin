"""Compare cooperative cross-mass construction with the scalar GPU oracle.

Distributed under the Boost Software License, Version 1.0.
"""

import argparse
import json
from pathlib import Path
from statistics import median

from gpu import cp, Space, gaussian
from graded_study import GradedMesh
from carrier import Carrier
from campaign import provenance


def check(n=32, p=4, grading=0.95, degree=None, factor=1, repeats=3):
    space = Space(GradedMesh(n, grading), p)
    options = dict(n=n * factor, degree=degree)
    # Warm both kernels and allocation paths before the repeated measurements.
    a = Carrier(space, builder="scalar", **options)
    b = Carrier(space, builder="warp", **options)
    samples = {"scalar": [], "warp": []}
    for repeat in range(repeats):
        for builder in ("scalar", "warp") if repeat % 2 == 0 else ("warp", "scalar"):
            value = Carrier(space, builder=builder, **options)
            samples[builder].append(value.build_seconds)
    assert bool(cp.all(a.source == b.source)) and bool(cp.all(a.target == b.target))
    block_error = float(cp.max(cp.abs(a.blocks - b.blocks)))
    assert block_error < 3e-11, block_error
    initial = space.project(gaussian)
    va = a.transport(0.17, flow="shear").apply(initial)
    vb = b.transport(0.17, flow="shear").apply(initial)
    solution_error = float(cp.max(cp.abs(va - vb)))
    assert solution_error < 3e-11, solution_error
    one = cp.zeros_like(initial)
    one[:, :, 0] = 1
    forward = float(cp.max(cp.abs(b.project(one) - 1)))
    backward = float(cp.max(cp.abs(b.adjoint(cp.ones((b.n * b.n, 1, b.nc))) - one)))
    assert max(forward, backward) < 5e-10, (forward, backward)
    random = cp.random.RandomState(42).normal(size=initial.shape)
    c = cp.random.RandomState(43).normal(size=(b.n * b.n, 1, b.nc))
    left = cp.sum(b.project(random) * c * b.mass[None, None, :])
    right = cp.sum(random * b.adjoint(c) * space.mesh.areas[:, None, None])
    adjoint = float(cp.abs(left - right))
    assert adjoint < 3e-12, adjoint
    return dict(
        n=n,
        p=p,
        grading=grading,
        degree=b.np - 1,
        factor=factor,
        samples=samples,
        medians={k: median(v) for k, v in samples.items()},
        block_error=block_error,
        solution_error=solution_error,
        forward_constant=forward,
        adjoint_constant=backward,
        adjoint_error=adjoint,
        scalar_attributes=a.builder.attributes,
        warp_attributes=b.builder.attributes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    cases = (
        [dict(n=8, p=4, grading=0.95), dict(n=8, p=2, degree=5, factor=2)]
        if args.small
        else [
            dict(n=n, p=p, grading=g)
            for n in (16, 32, 64)
            for p in (1, 2, 3, 4)
            for g in (0.0, 0.95)
        ]
        + [
            dict(n=8, p=p, degree=q, factor=f)
            for p, q, f in ((0, 0, 1), (1, 5, 2), (2, 3, 2), (3, 5, 1), (4, 2, 2))
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = provenance()
    with args.output.open("w") as f:
        for case in cases:
            result = check(**case, repeats=1 if args.small else 3)
            f.write(
                json.dumps(dict(status="ok", provenance=meta, result=result)) + "\n"
            )
            f.flush()
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in ("n", "p", "grading", "medians", "block_error")
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
