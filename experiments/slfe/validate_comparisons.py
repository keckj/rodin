"""Focused checks for the direct DG and graded coupled GPU comparisons.

Distributed under the Boost Software License, Version 1.0.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Space
from dg_study import validate
from graded_study import GradedMesh
from fourier_carrier import FourierCarrier
from coupled_benchmark import evolve
from campaign import provenance


def graded_projection():
    space = Space(GradedMesh(8, 0.95), 4)
    coupling = FourierCarrier(space, modes=39)
    nv = space.mesh.cells * space.nb
    eye = cp.ascontiguousarray(
        cp.eye(nv).reshape(space.mesh.cells, space.nb, nv).transpose(0, 2, 1)
    )
    mass = cp.repeat(space.mesh.areas, space.nb)
    matrix = coupling.project(eye).reshape(coupling.n**2, nv)
    norm = float(
        cp.linalg.svd(matrix / (coupling.n * cp.sqrt(mass)[None, :]), compute_uv=False)[
            0
        ]
    )
    assert norm < 1 + 3e-11, norm
    one = cp.zeros((space.mesh.cells, 1, space.nb))
    one[:, :, 0] = 1
    forward = float(cp.max(cp.abs(coupling.project(one) - 1)))
    reverse = float(
        cp.max(cp.abs(coupling.adjoint(cp.ones((coupling.n, coupling.n, 1))) - one))
    )
    assert max(forward, reverse) < 3e-11, (forward, reverse)
    return dict(
        norm=norm,
        forward_constant_error=forward,
        reverse_constant_error=reverse,
        area_ratio=space.mesh.area_ratio,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    result = dict(
        status="ok",
        provenance=provenance(),
        dg=[
            validate(p, flux)
            for p in ((2,) if args.small else (1, 2, 3, 4))
            for flux in ("central", "upwind")
        ],
        coupled=evolve(n=8, grading=0.95, steps=2),
        direct=evolve(n=8, grading=0.95, method="dg", cfl_scale=4.0),
    )
    if not args.small:
        result["graded_projection"] = graded_projection()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="ok", dg_checks=len(result["dg"]))))


if __name__ == "__main__":
    main()
