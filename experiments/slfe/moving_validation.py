"""Dense CUDA validation of exact two-axis moving-grid projection.

Distributed under the Boost Software License, Version 1.0.
Checks mass, constants, contraction, and the mass adjoint for different grids.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Space, Mesh
from carrier import Carrier
from tensor_grid import TensorGrid
from adaptive_study import partitions
from campaign import provenance


def validate(p):
    space = Space(Mesh(4), p)
    carrier = Carrier(space, edges=partitions(4, 0.8, cp.asarray([0.35, 0.45])))
    first = TensorGrid(carrier, carrier.edges)
    second = TensorGrid(carrier, partitions(4, 0.6, cp.asarray([0.81, 0.02])))
    n = first.n**2 * first.nc
    eye = cp.ascontiguousarray(
        cp.eye(n).reshape(first.n**2, first.nc, n).transpose(0, 2, 1)
    )
    matrix = lambda v: v.transpose(0, 2, 1).reshape(n, n)
    forward = matrix(second.remap(first, eye))
    reverse = matrix(first.remap(second, eye))
    m0, m1 = first.mass.ravel(), second.mass.ravel()
    adjoint = float(cp.max(cp.abs(reverse - forward.T * m1[None, :] / m0[:, None])))
    sv = float(
        cp.linalg.svd(
            cp.sqrt(m1)[:, None] * forward / cp.sqrt(m0)[None, :], compute_uv=False
        )[0]
    )
    constants = float(cp.max(cp.abs(forward.sum(axis=1) - 1)))
    mass = float(cp.max(cp.abs(m1 @ forward - m0)))
    assert max(adjoint, constants, mass) < 3e-11 and sv < 1 + 3e-11
    identity = float(cp.max(cp.abs(matrix(first.remap(first, eye)) - cp.eye(n))))
    assert identity < 3e-11
    return dict(
        p=p,
        singular_max=sv,
        adjoint_error=adjoint,
        constant_error=constants,
        mass_error=mass,
        identity_error=identity,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [validate(p) for p in (1, 2, 4)]
    args.output.write_text(
        json.dumps(dict(results=results, provenance=provenance()), indent=2) + "\n"
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
