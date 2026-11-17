"""GPU exponential-DG comparator with independent matrix and PDE references.

Distributed under the Boost Software License, Version 1.0.
The reference quadrature is built once on GPU and shared by the parameter sweep.
Krylov truncation is selected empirically here; no automatic tolerance is claimed.
"""

import argparse
import json
from pathlib import Path
import time

from cupyx.scipy.linalg import expm

from gpu import cp, Mesh, Space, quadrature
from graded_study import GradedMesh
from eulerian_dg import EulerianDG
from arnoldi_transport import ArnoldiTransport
from moving_study import profile_family
from day_study import reference_feet
from campaign import provenance


def validate():
    records = []
    for degree in (1, 2, 4):
        space = Space(Mesh(4, jitter=0.18, seed=817), degree)
        for flux in ("central", "upwind"):
            flow = EulerianDG(space, flux=flux)
            solver = ArnoldiTransport(flow, dimension=48)
            size = solver.size
            identity = cp.ascontiguousarray(
                cp.eye(size)
                .reshape(space.mesh.cells, space.nb, size)
                .transpose(0, 2, 1)
            )
            matrix = flow.rhs(identity).transpose(0, 2, 1).reshape(size, size)
            mass = cp.repeat(space.mesh.areas, space.nb)
            matrix *= cp.sqrt(mass[:, None] / mass[None, :])
            initial = cp.random.RandomState(317).normal(
                size=(space.mesh.cells, 2, space.nb)
            )
            value = solver.step(initial, 0.001)
            weighted = (initial * solver.root_mass).transpose(0, 2, 1).reshape(size, 2)
            exact = expm(0.001 * matrix) @ weighted
            actual = (value * solver.root_mass).transpose(0, 2, 1).reshape(size, 2)
            error = float(cp.linalg.norm(actual - exact) / cp.linalg.norm(exact))
            mass_error = float(cp.max(cp.abs(space.mass(value) - space.mass(initial))))
            ratio = float(cp.max(space.energy(value) / space.energy(initial)))
            assert error < 1e-9 and mass_error < 1e-11 and ratio < 1 + 1e-10, (
                error,
                mass_error,
                ratio,
            )
            records.append(
                dict(
                    degree=degree,
                    flux=flux,
                    relative_error=error,
                    mass_error=mass_error,
                    energy_ratio=ratio,
                )
            )
    return records


class ErrorOracle:
    def __init__(self, space, profile, end):
        self.space = space
        q = quadrature(20, subdivisions=2)
        self.basis = space.basis(q[:, :2])
        self.weights = 2 * space.mesh.areas[:, None] * q[None, :, 2]
        # Limit temporary characteristic arrays while retaining the final oracle.
        pieces = []
        for first in range(0, space.mesh.cells, 4096):
            triangles = space.mesh.triangles[first : first + 4096]
            points = triangles[:, None, 0] + q[None, :, :1] * (
                triangles[:, None, 1] - triangles[:, None, 0]
            )
            points += q[None, :, 1:2] * (triangles[:, None, 2] - triangles[:, None, 0])
            pieces.append(profile(reference_feet(points, -end, 512), 0))
        self.reference = cp.concatenate(pieces)
        self.best = cp.einsum("cqf,qb,q->cfb", self.reference, self.basis, 2 * q[:, 2])

    def errors(self, coefficients):
        actual = cp.einsum("cfb,qb->cqf", coefficients, self.basis)
        difference = actual - self.reference
        return dict(
            l2=cp.sqrt(
                cp.sum(difference**2 * self.weights[..., None], axis=(0, 1))
            ).tolist(),
            l1=cp.sum(
                cp.abs(difference) * self.weights[..., None], axis=(0, 1)
            ).tolist(),
            linf_sampled=cp.max(cp.abs(difference), axis=(0, 1)).tolist(),
            min_sampled=cp.min(actual, axis=(0, 1)).tolist(),
            max_sampled=cp.max(actual, axis=(0, 1)).tolist(),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nfe", type=int, default=128)
    args = parser.parse_args()
    meta = provenance()
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(dict(results=validate(), provenance=meta), indent=2) + "\n"
    )
    space = Space(GradedMesh(args.nfe, 0.95), 4)
    profile = profile_family(0.01, 1)
    initial = space.project(profile, degree=20)
    oracle = ErrorOracle(space, profile, 0.25)
    floor = oracle.errors(oracle.best)["l2"]
    with args.output.open("w") as stream:
        for flux in ("central", "upwind"):
            cp.cuda.Stream.null.synchronize()
            before = time.perf_counter()
            flow = EulerianDG(space, flux=flux)
            cp.cuda.Stream.null.synchronize()
            assembly = time.perf_counter() - before
            for dimension in (48, 96):
                for steps in (1, 4, 16, 64):
                    print(
                        json.dumps(dict(flux=flux, dimension=dimension, steps=steps)),
                        flush=True,
                    )
                    solver = ArnoldiTransport(flow, dimension)
                    cp.cuda.Stream.null.synchronize()
                    start = time.perf_counter()
                    solver.step(initial, 0.25 / steps)
                    cp.cuda.Stream.null.synchronize()
                    setup = assembly + time.perf_counter() - start
                    solver.history.clear()
                    value = initial.copy()
                    start = time.perf_counter()
                    for _ in range(steps):
                        value = solver.step(value, 0.25 / steps)
                    cp.cuda.Stream.null.synchronize()
                    elapsed = time.perf_counter() - start
                    ratio = float(cp.max(space.energy(value) / space.energy(initial)))
                    mass_error = float(
                        cp.max(cp.abs(space.mass(value) - space.mass(initial)))
                    )
                    assert ratio < 1 + 1e-8 and mass_error < 1e-9
                    errors = oracle.errors(value)
                    result = dict(
                        nfe=args.nfe,
                        fields=1,
                        end=0.25,
                        flux=flux,
                        dimension=dimension,
                        steps=steps,
                        setup_seconds=setup,
                        evolution_seconds=elapsed,
                        accepted=max(errors["l2"]) <= 0.001,
                        mass_error=mass_error,
                        energy_ratio=ratio,
                        best_fe_l2=floor,
                        generator_applications=sum(
                            h["dimension"] for h in solver.history
                        ),
                        residual_indicator_max=max(
                            h["residual_indicator"] for h in solver.history
                        ),
                        **errors
                    )
                    stream.write(
                        json.dumps(dict(result=result, provenance=meta)) + "\n"
                    )
                    stream.flush()
                    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
