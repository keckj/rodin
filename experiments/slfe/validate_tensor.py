"""Independent dense checks for graded coordinate maps and uniform regression.

Distributed under the Boost Software License, Version 1.0.
The uniform oracle is the published implementation before nonuniform support.
All numerical arrays, decompositions and reductions remain on CUDA.
"""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from types import ModuleType

from gpu import cp, ROOT, Space
from graded_study import GradedMesh
from carrier import Carrier
from tensor_study import partitions, validate
from tensor_cellular import TensorCellular
from campaign import provenance


def uniform_regression():
    revision = "753756f193d"
    with tempfile.TemporaryDirectory(prefix="slfe-uniform-oracle-") as folder:
        folder = Path(folder)
        for name in (
            "carrier.py",
            "carrier.cu",
            "basis.cuh",
            "transfer.cu",
            "carrier_polynomials.json",
        ):
            data = subprocess.check_output(
                ["git", "show", f"{revision}:experiments/slfe/{name}"], cwd=ROOT
            )
            (folder / name).write_bytes(data)
        old = ModuleType("uniform_carrier_oracle")
        exec(
            compile(
                (folder / "carrier.py").read_text(), str(folder / "carrier.py"), "exec"
            ),
            old.__dict__,
        )
        old.HERE = folder
        records = []
        for n, p in ((5, 2), (8, 4)):
            s = Space(GradedMesh(8, 0.7), p)
            a, b = old.Carrier(s, n=n), Carrier(s, n=n)
            random = cp.random.RandomState(82)
            v = random.normal(size=(s.mesh.cells, 3, s.nb))
            w = random.normal(size=(n * n, 3, a.nc))
            differences = dict(
                blocks=float(cp.max(cp.abs(a.blocks - b.blocks))),
                project=float(cp.max(cp.abs(a.project(v) - b.project(v)))),
                adjoint=float(cp.max(cp.abs(a.adjoint(w) - b.adjoint(w)))),
            )
            for flow in ("translation", "shear"):
                differences[flow] = float(
                    cp.max(
                        cp.abs(
                            a.transport(0.17, flow).apply(v)
                            - b.transport(0.17, flow).apply(v)
                        )
                    )
                )
            assert max(differences.values()) < 1e-11, differences
            records.append(dict(n=n, p=p, differences=differences))
    return records


def coordinate_validation(p=2):
    s = Space(GradedMesh(4, 0.6), p)
    c = Carrier(s, edges=partitions(4, 0.9))
    n = c.n**2 * c.nc
    eye = cp.ascontiguousarray(cp.eye(n).reshape(c.n**2, c.nc, n).transpose(0, 2, 1))
    mass = c.mass.ravel()
    records = []
    for dt in (0.0, 0.005, 0.71, -0.71):
        flow = TensorCellular(c, dt, grading=0.9)
        coordinates = []
        for grid, forward, adjoint in flow.coordinate_maps:
            f = forward.apply_carrier(eye).transpose(0, 2, 1).reshape(n, n)
            a = adjoint.apply_carrier(eye).transpose(0, 2, 1).reshape(n, n)
            h = cp.diff(grid.edges, axis=1)
            m = (
                (h[1, :, None] * h[0, None, :]).ravel()[:, None]
                * (c.weights[:, None] * c.weights[None, :]).ravel()
            ).ravel()
            difference = float(cp.max(cp.abs(a - f.T * m[None, :] / mass[:, None])))
            sv = float(
                cp.linalg.svd(
                    cp.sqrt(m)[:, None] * f / cp.sqrt(mass)[None, :], compute_uv=False
                )[0]
            )
            assert difference < 3e-11 and sv < 1 + 3e-11, (difference, sv)
            coordinates.append(dict(adjoint_error=difference, singular_max=sv))
        full = flow.apply_carrier(eye).transpose(0, 2, 1).reshape(n, n)
        sv = float(
            cp.linalg.svd(
                cp.sqrt(mass)[:, None] * full / cp.sqrt(mass)[None, :], compute_uv=False
            )[0]
        )
        constant = float(cp.max(cp.abs(full.sum(axis=1) - 1)))
        conservation = float(cp.max(cp.abs(mass @ full - mass)))
        assert sv < 1 + 4e-11 and max(constant, conservation) < 4e-11
        if dt == 0:
            assert float(cp.max(cp.abs(full - cp.eye(n)))) < 4e-11
        records.append(
            dict(
                dt=dt,
                singular_max=sv,
                constant_error=constant,
                mass_error=conservation,
                coordinates=coordinates,
            )
        )
    return dict(p=p, records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(
        uniform=uniform_regression(),
        projection_and_shear=[validate(p) for p in (1, 2, 4)],
        cellular=[coordinate_validation(p) for p in (2, 4)],
        provenance=provenance(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Uniform regression, graded projections, shears and cellular adjoints passed."
    )


if __name__ == "__main__":
    main()
