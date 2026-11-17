"""GPU controls for positive common-quadrature residual energies.

Distributed under the Boost Software License, Version 1.0.
Random fields compare with the independent cross-mass bilinear identity;
nearly equal constants compare with their exact scalar difference instead.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from carrier import Carrier
from residual_energy import ResidualEnergy
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    random = cp.random.RandomState(1729)
    for degree in (1, 2, 4):
        space = Space(Mesh(4, jitter=0.17, seed=916), degree)
        for shifted in (False, True):
            line = cp.linspace(0, 1, 5)
            edges = (
                cp.stack(
                    (
                        line + 0.03 * cp.sin(2 * cp.pi * line) + 0.07,
                        line - 0.02 * cp.sin(2 * cp.pi * line) - 0.04,
                    )
                )
                if shifted
                else None
            )
            c = Carrier(space, n=4, edges=edges, builder="tiled")
            diagnostic = ResidualEnergy(c)
            assert float(cp.min(diagnostic.quadrature[:, 2])) > 0
            fe = random.normal(size=(space.mesh.cells, 4, space.nb))
            auxiliary = random.normal(size=(c.n**2, 4, c.nc))
            direct = diagnostic.apply(fe, auxiliary)
            cross = cp.sum(
                fe * c.adjoint(auxiliary) * space.mesh.areas[:, None, None], axis=(0, 2)
            )
            oracle = space.energy(fe) + c.energy(auxiliary) - 2 * cross
            relative_error = float(cp.max(cp.abs(direct - oracle) / oracle))
            fe_error = float(
                cp.max(
                    cp.abs(
                        diagnostic.apply(fe, cp.zeros_like(auxiliary))
                        / space.energy(fe)
                        - 1
                    )
                )
            )
            auxiliary_error = float(
                cp.max(
                    cp.abs(
                        diagnostic.apply(cp.zeros_like(fe), auxiliary)
                        / c.energy(auxiliary)
                        - 1
                    )
                )
            )
            assert max(relative_error, fe_error, auxiliary_error) < 1e-11
            constant = cp.zeros_like(fe)
            constant[:, :, 0] = 1
            cancellation = []
            for perturbation in (0.0, 1e-4, 1e-8, 1e-12):
                nearby = cp.full_like(auxiliary, 1 + perturbation)
                exact = (nearby[0, 0, 0] - 1) ** 2
                positive = diagnostic.apply(constant, nearby)
                difference = (
                    space.energy(constant)
                    + c.energy(nearby)
                    - 2
                    * cp.sum(
                        constant * c.adjoint(nearby) * space.mesh.areas[:, None, None],
                        axis=(0, 2),
                    )
                )
                error = float(cp.max(cp.abs(positive - exact)))
                assert error < 0.003 * float(exact) + 1e-27, (
                    degree,
                    shifted,
                    perturbation,
                    error,
                    float(exact),
                )
                cancellation.append(
                    dict(
                        perturbation=perturbation,
                        exact=float(exact),
                        positive=positive.tolist(),
                        subtractive=difference.tolist(),
                    )
                )
            record = dict(
                degree=degree,
                shifted=shifted,
                random_relative_error=relative_error,
                fe_relative_error=fe_error,
                auxiliary_relative_error=auxiliary_error,
                cancellation=cancellation,
            )
            records.append(record)
            print(json.dumps(record), flush=True)
            args.output.write_text(
                json.dumps(dict(results=records, provenance=provenance()), indent=2)
                + "\n"
            )


if __name__ == "__main__":
    main()
