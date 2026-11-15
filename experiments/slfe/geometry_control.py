"""Isolate Jacobian inconsistency from transfer quadrature error on GPU.

For a constant density and a piecewise-affine map with J_K>0, the exact
pre-projection squared L2 error against 1 is sum_K |K| (J_K-1)^2/J_K.
Orthogonal projection can only decrease it. This is an independent oracle.
Distributed under the Boost Software License, Version 1.0.
"""
import argparse
import json
from pathlib import Path

from gpu import cp, Mesh, Space, Transfer, constant
from flows import shear, compressible, compressible_jacobian, reversible


def run():
    records = []
    for n in [8, 16, 32, 64, 128, 256]:
        for jitter in [0., .18]:
            mesh = Mesh(n, jitter=jitter)
            moved = mesh.mapped(shear, 0, 1/16)
            e, f = moved[:, 1]-moved[:, 0], moved[:, 2]-moved[:, 0]
            area = (e[:, 0]*f[:, 1]-e[:, 1]*f[:, 0])/2
            jac = area/mesh.areas
            bound = float(cp.sqrt(cp.sum(mesh.areas*(jac-1)**2/jac)))
            for p in [0, 1, 2, 3]:
                space = Space(mesh, p)
                one = cp.zeros((mesh.cells, 1, space.nb), dtype=cp.float64)
                one[:, :, 0] = 1.
                transfer = Transfer(space, moved)
                actual = space.errors(transfer.apply(one), constant, 1/16)["l2"][0]
                assert actual <= bound+1e-11, (n, p, jitter, actual, bound)
                if jitter == 0.:
                    assert actual < 1e-11, (n, p, actual)
                records.append({"n": n, "p": p, "jitter": jitter, "dt": 1/16,
                                "jacobian_min": float(cp.min(jac)), "jacobian_max": float(cp.max(jac)),
                                "exact_before_projection_l2": bound, "after_projection_l2": actual})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = run()
    args.output.write_text(json.dumps({"passed": True, "records": rows}, indent=2)+"\n")
    print(f"Passed {len(rows)} independent geometric controls")
