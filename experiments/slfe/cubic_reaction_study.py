"""GPU advection with a genuinely nonlinear reaction of the full memory.

Distributed under the Boost Software License, Version 1.0.
For u_t+b.grad(u)=-kappa*r(x)*u^3, the characteristic solution is
u0/sqrt(1+2*kappa*u0^2*integral(r)). The independent trajectory kernel
already integrates r; no numerical nonlinear ODE reference is required.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Space, launch, quadrature
from graded_study import GradedMesh
from moving_study import profile_family
from carrier import Carrier
from tensor_cellular import TensorCellular
from memory_transport import MemoryTransport
from reaction_study import reference_kernel
from quadrature_moments import QuadratureMoments
from residual_energy import ResidualEnergy
from krylov_study import ErrorOracle
from campaign import provenance


class CubicFE:
    def __init__(self, space, duration, subdivisions=1):
        self.quadrature = quadrature(20, subdivisions)
        self.basis = space.basis(self.quadrature[:, :2])
        x = space.mesh.points(self.quadrature)[..., 0]
        self.coefficient = 2 * duration * (1 + 0.5 * cp.sin(2 * cp.pi * x))

    def apply(self, value):
        sampled = cp.einsum("cfb,qb->cqf", value, self.basis)
        transformed = sampled / cp.sqrt(1 + self.coefficient[..., None] * sampled**2)
        return cp.ascontiguousarray(
            cp.einsum(
                "cqf,qb,q->cfb", transformed, self.basis, 2 * self.quadrature[:, 2]
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--steps", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.01, 1, 100])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("fe-only", "fe-reset", "fe-common", "ambient-moments"),
        default=["fe-only", "ambient-moments"],
    )
    parser.add_argument("--subdivisions", type=int, default=1)
    args = parser.parse_args()
    meta = provenance()
    with args.output.open("w") as stream:
        for n in args.resolutions:
            space = Space(GradedMesh(n, 0.95), 4)
            profile = profile_family(0.01, 1)
            initial = space.project(profile, degree=20)
            carrier = Carrier(space, n=2 * n, builder="tiled")
            positive = ResidualEnergy(carrier)
            for strength in args.strengths:

                def exact(points, substeps=512):
                    flat = cp.ascontiguousarray(points.reshape(-1, 2))
                    traced = cp.empty((len(flat), 3))
                    launch(
                        reference_kernel(),
                        len(flat),
                        (flat, traced, len(flat), 0.25, substeps),
                    )
                    original = profile(traced[:, :2], 0)
                    return (
                        original
                        / cp.sqrt(1 - 2 * strength * traced[:, 2, None] * original**2)
                    ).reshape(*points.shape[:-1], 1)

                probe = cp.random.RandomState(13).uniform(size=(4096, 2))
                refined = exact(probe, 1024)
                ref_error = float(
                    cp.max(cp.abs(exact(probe) - refined)) / cp.max(cp.abs(refined))
                )
                assert ref_error < 1e-7
                oracle = ErrorOracle(space, profile, 0.25, reference=exact)
                norm = float(
                    cp.sqrt(cp.sum(oracle.reference**2 * oracle.weights[..., None]))
                )
                floor = oracle.errors(oracle.best)["l2"][0] / norm
                for steps in args.steps:
                    flow = TensorCellular(carrier, 0.25 / steps, grading=0.0)
                    duration = strength * 0.25 / (2 * steps)
                    local = CubicFE(space, duration, args.subdivisions)
                    complete = QuadratureMoments(
                        carrier, duration, subdivisions=args.subdivisions
                    )
                    for method in args.methods:
                        print(
                            json.dumps(
                                dict(n=n, steps=steps, strength=strength, method=method)
                            ),
                            flush=True,
                        )
                        state = MemoryTransport(carrier, initial)
                        # Warm the common-quadrature kernel outside timing.
                        if method in ("fe-common", "ambient-moments"):
                            complete.moments(
                                state.anchor,
                                state.auxiliary - carrier.project(state.anchor),
                            )
                        energy0 = float(space.energy(initial)[0])
                        maximum = 1.0
                        cp.cuda.Stream.null.synchronize()
                        begin = time.perf_counter()
                        for _ in range(steps):
                            for half in (0, 1):
                                if method == "ambient-moments":
                                    complete.apply(state)
                                elif method == "fe-common":
                                    state.apply_physics(
                                        lambda v: complete.moments(
                                            v, cp.zeros_like(state.auxiliary)
                                        )[0]
                                    )
                                else:
                                    state.apply_physics(
                                        local.apply,
                                        hidden_decay=(
                                            0.0 if method == "fe-reset" else None
                                        ),
                                    )
                                if half == 0:
                                    state.advect(flow.apply_carrier)
                            maximum = max(maximum, float(state.energy()[0]) / energy0)
                        cp.cuda.Stream.null.synchronize()
                        elapsed = time.perf_counter() - begin
                        actual_energy = float(
                            positive.apply(
                                state.anchor,
                                carrier.project(state.anchor) - state.auxiliary,
                            )[0]
                        )
                        assert maximum <= 1 + 1e-8 and actual_energy <= energy0 * (
                            1 + 1e-8
                        )
                        errors = oracle.errors(state.value)
                        result = dict(
                            nfe=n,
                            naux=2 * n,
                            steps=steps,
                            strength=strength,
                            end=0.25,
                            method=method,
                            quadrature_subdivisions=args.subdivisions,
                            relative_l2=errors["l2"][0] / norm,
                            reference_norm=norm,
                            relative_fe_floor=floor,
                            reference_refinement_error=ref_error,
                            maximum_energy_ratio=maximum,
                            positive_energy=actual_energy,
                            evolution_seconds=elapsed,
                            **errors
                        )
                        stream.write(
                            json.dumps(dict(result=result, provenance=meta)) + "\n"
                        )
                        stream.flush()
                        print(json.dumps(result), flush=True)
                del oracle
            del carrier, space
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
