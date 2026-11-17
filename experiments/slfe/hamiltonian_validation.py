"""GPU adjoint, contraction and characteristic controls for general mode charts.

Distributed under the Boost Software License, Version 1.0.
Dense transport matrices are built only as independent small CUDA oracles.
"""

import argparse
import json
import math
from pathlib import Path

from gpu import cp, HERE, Mesh, Space, launch
from carrier import Carrier
from tensor_hamiltonian import (
    TensorHamiltonian,
    ComposedHamiltonian,
    integer_chart,
    streamfunction_modes,
)
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vectors = [
        (1, 0),
        (0, 1),
        (1, 1),
        (1, -1),
        (2, 1),
        (1, 2),
        (3, 2),
        (-3, 2),
        (3, -2),
        (0, -1),
        (2, 2),
    ]
    results = dict(provenance=provenance(), controls=[], characteristics=[])
    reference = cp.RawKernel(
        (HERE / "hamiltonian_reference.cu").read_text(), "hamiltonian_reference"
    )
    rng = cp.random.RandomState(621)
    for degree in (1, 2, 4):
        space = Space(Mesh(4, jitter=0.15, seed=819), degree)
        carrier = Carrier(space, n=4, builder="tiled")
        mass = cp.broadcast_to(carrier.mass, (carrier.n**2, carrier.nc))
        a, b = [rng.normal(size=(carrier.n**2, 2, carrier.nc)) for _ in range(2)]
        ones = cp.ones_like(a)
        for wave in vectors:
            coefficients = cp.asarray([[0.013, -0.007]])
            flow = TensorHamiltonian(carrier, [wave], coefficients, 0.17)
            operators, _ = flow.charts[0]
            ca, ctb = flow.transform(a, operators), flow.transform(
                b, operators, adjoint=True
            )
            adjoint_error = float(
                cp.max(
                    cp.abs(cp.sum((ca * b - a * ctb) * mass[:, None, :], axis=(0, 2)))
                )
            )
            actual = flow.apply_carrier(a)
            ratio = float(cp.max(carrier.energy(actual) / carrier.energy(a)))
            mass_error = float(
                cp.max(cp.abs(cp.sum((actual - a) * mass[:, None, :], axis=(0, 2))))
            )
            constant_error = float(cp.max(cp.abs(flow.apply_carrier(ones) - ones)))
            zero = TensorHamiltonian(carrier, [wave], coefficients, 0.0).apply_carrier(
                a
            )
            identity_error = float(cp.max(cp.abs(zero - a)))
            assert (
                max(adjoint_error, mass_error, constant_error, identity_error) < 1e-11
            )
            assert ratio <= 1 + 1e-11
            record = dict(
                degree=degree,
                wave=wave,
                chart=flow.matrices[0],
                adjoint_error=adjoint_error,
                energy_ratio=ratio,
                mass_error=mass_error,
                constant_error=constant_error,
                zero_time_identity_error=identity_error,
            )
            results["controls"].append(record)
            print(json.dumps(record), flush=True)
        flow = TensorHamiltonian(
            carrier,
            vectors,
            cp.tile(cp.asarray([[0.013, -0.007]]), (len(vectors), 1)),
            0.17,
        )
        nw = carrier.n**2 * carrier.nc
        identity = cp.ascontiguousarray(
            cp.eye(nw).reshape(carrier.n**2, carrier.nc, nw).transpose(0, 2, 1)
        )
        matrix = flow.apply_carrier(identity).transpose(0, 2, 1).reshape(nw, nw)
        matrix *= cp.sqrt(mass.ravel()[:, None] / mass.ravel()[None, :])
        norm = float(cp.linalg.svd(matrix, compute_uv=False)[0])
        assert norm <= 1 + 1e-11, norm
        results["controls"].append(dict(degree=degree, dense_transport_norm=norm))
        for scheme in ("triple4", "suzuki4", "yoshida6"):
            coefficients = cp.tile(cp.asarray([[0.013, -0.007]]), (len(vectors), 1))
            composed = ComposedHamiltonian(carrier, vectors, coefficients, 0.17, scheme)
            matrix = composed.apply_carrier(identity).transpose(0, 2, 1).reshape(nw, nw)
            matrix *= cp.sqrt(mass.ravel()[:, None] / mass.ravel()[None, :])
            norm = float(cp.linalg.svd(matrix, compute_uv=False)[0])
            constant_error = float(cp.max(cp.abs(composed.apply_carrier(ones) - ones)))
            actual = composed.apply_carrier(a)
            mass_error = float(
                cp.max(cp.abs(cp.sum((actual - a) * mass[:, None, :], axis=(0, 2))))
            )
            zero = ComposedHamiltonian(
                carrier, vectors, coefficients, 0.0, scheme
            ).apply_carrier(a)
            identity_error = float(cp.max(cp.abs(zero - a)))
            independent = a.copy()
            for weight in composed.weights:
                independent = TensorHamiltonian(
                    carrier, vectors, coefficients, 0.17 * weight
                ).apply_carrier(independent)
            shared_map_error = float(
                cp.max(cp.abs(independent - composed.apply_carrier(a)))
            )
            assert shared_map_error < 1e-13
            assert norm <= 1 + 1e-11, norm
            assert max(constant_error, mass_error, identity_error) < 1e-11
            results["controls"].append(
                dict(
                    degree=degree,
                    scheme=scheme,
                    shared_map_error=shared_map_error,
                    dense_transport_norm=norm,
                    constant_error=constant_error,
                    mass_error=mass_error,
                    zero_time_identity_error=identity_error,
                )
            )
        scalar = rng.normal(size=(space.mesh.cells, 1, space.nb))
        coarse = streamfunction_modes(space, scalar, vectors)
        fine = streamfunction_modes(space, scalar, vectors, subdivisions=2)
        moment_error = float(cp.max(cp.abs(coarse - fine)))
        assert moment_error < 1e-9, moment_error
        results["controls"].append(
            dict(degree=degree, fourier_quadrature_error=moment_error)
        )
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    points = rng.uniform(size=(4096, 2))
    for k, l in vectors:
        modes = cp.asarray([[k, l, 0.013, -0.007]], dtype=cp.float64)
        angle = 2 * cp.pi * (k * points[:, 0] + l * points[:, 1])
        derivative = 2 * cp.pi * (-0.013 * cp.sin(angle) - 0.007 * cp.cos(angle))
        expected = points - 0.37 * cp.stack((l * derivative, -k * derivative), axis=1)
        actual = cp.empty_like(points)
        launch(
            reference, len(points), (points, actual, len(points), modes, 1, -0.37, 512)
        )
        error = float(cp.max(cp.abs(actual - expected)))
        assert error < 1e-11, error
        results["characteristics"].append(dict(wave=(k, l), analytic_foot_error=error))
    # Symbolic setup also covers integer directions beyond the PDE pilot.
    for k in range(-7, 8):
        for l in range(-7, 8):
            if math.gcd(k, l) == 1:
                integer_chart(k, l)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
