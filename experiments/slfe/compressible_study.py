"""Conservative forward remapping for an exact compressible characteristic map.

Distributed under the Boost Software License, Version 1.0.
The prescribed periodic velocity is (kappa sin(2 pi x), 0). Source intervals
are split at target-edge preimages before quadrature. All numerical work is
CUDA; conservation and physical L2 growth are checked separately.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Space, Mesh
from carrier import Carrier
from tensor_projection import TensorProjection
from memory_transport import MemoryTransport
from product_study import profile
from campaign import provenance


def characteristic(x, time, kappa=0.2):
    period = cp.floor(x)
    angle = cp.pi * (x - period)
    scale = cp.exp(cp.asarray(2 * cp.pi * kappa * time))
    return period + cp.arctan2(scale * cp.sin(angle), cp.cos(angle)) / cp.pi


def jacobian(x, time, kappa=0.2):
    scale = cp.exp(cp.asarray(2 * cp.pi * kappa * time))
    return scale / (cp.cos(cp.pi * x) ** 2 + scale**2 * cp.sin(cp.pi * x) ** 2)


def exact(points, time):
    feet = points.copy()
    feet[..., 0] = characteristic(points[..., 0], -time)
    # Conservative density includes the inverse Jacobian. Dropping this
    # factor would solve the advective equation, which is different here.
    return profile(feet) * jacobian(points[..., 0], -time)


def maps(duration):
    return (
        (lambda x: characteristic(x, duration), lambda x: characteristic(x, -duration)),
        (lambda x: x, lambda x: x),
    )


def validate():
    points = cp.random.RandomState(743).uniform(0, 1, (8192, 2))
    inverse = float(
        cp.max(
            cp.abs(
                characteristic(characteristic(points[..., 0], 0.7), -0.7)
                - points[..., 0]
            )
        )
    )
    assert inverse < 1e-14, inverse
    space = Space(Mesh(4, jitter=0.18, seed=817), 4)
    carrier = Carrier(space, n=8)
    records = []
    for duration in (0.0, 0.1, 0.5, 1.0, -1.0):
        flow = TensorProjection(carrier, carrier, maps(duration), quadrature_order=16)
        refined = TensorProjection(
            carrier, carrier, maps(duration), quadrature_order=32
        )
        errors, norms, masses = [], [], []
        for axis in (0, 1):
            matrix = flow.matrices[axis].toarray()
            error = float(cp.max(cp.abs(matrix - refined.matrices[axis].toarray())))
            mass = cp.repeat(cp.diff(carrier.edges[axis]), carrier.np) * cp.tile(
                carrier.weights, carrier.n
            )
            weighted = cp.sqrt(mass[:, None] / mass[None, :]) * matrix
            norm = float(cp.linalg.svd(weighted, compute_uv=False)[0])
            mass_error = float(cp.max(cp.abs(mass @ matrix - mass)))
            assert error < 1e-11 and mass_error < 1e-13, (error, mass_error)
            assert norm <= float(cp.exp(cp.pi * 0.2 * abs(duration))) + 1e-11
            errors.append(error)
            norms.append(norm)
            masses.append(mass_error)
        if duration == 0:
            identity = float(
                cp.max(
                    cp.abs(flow.matrices[0].toarray() - cp.eye(carrier.n * carrier.np))
                )
            )
            assert identity < 1e-12, identity
        records.append(
            dict(
                duration=duration,
                quadrature_errors=errors,
                matrix_norms=norms,
                mass_errors=masses,
            )
        )
    random = cp.random.RandomState(923).normal(size=(carrier.n**2, 3, carrier.nc))
    original = TensorProjection(carrier, carrier).apply_carrier(random)
    assert float(cp.max(cp.abs(original - random))) < 1e-11
    return dict(inverse_error=inverse, records=records)


def evolve(nfe=64, naux=64, steps=16, method="memory", end=1.0):
    space = Space(Mesh(nfe, jitter=0.18, seed=817), 4)
    initial = space.project(lambda points, t: exact(points, 0), degree=20)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    carrier = Carrier(space, n=naux)
    flow = TensorProjection(carrier, carrier, maps(end / steps), quadrature_order=16)
    state = MemoryTransport(carrier, initial)
    energy0 = state.energy()
    mass0 = space.mass(initial)
    # CuPy 13's CSR matrix product uses a cuBLAS layout conversion that
    # cannot be captured. The same numerical GPU path runs without a graph.
    flow.apply_carrier(state.auxiliary)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    if method == "memory":
        transported = state.auxiliary
        for _ in range(steps):
            transported = flow.apply_carrier(transported)
        state.advect(lambda w: transported)
    elif method == "reset":
        for _ in range(steps):
            state.advect(flow.apply_carrier)
            state.restart(carrier)
    else:
        raise ValueError("Expected memory or reset")
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    energy_ratio = float(cp.max(state.energy() / energy0))
    physical_bound = float(cp.exp(cp.asarray(2 * cp.pi * 0.2 * abs(end))))
    mass_error = float(cp.max(cp.abs(space.mass(state.value) - mass0)))
    assert energy_ratio <= physical_bound + 1e-9 and mass_error < 1e-10
    errors = space.errors(
        state.value, lambda p, t: exact(p, end), 0, degree=20, subdivisions=2
    )
    projected = space.project(lambda p, t: exact(p, end), degree=20)
    floor = space.errors(
        projected, lambda p, t: exact(p, end), 0, degree=20, subdivisions=2
    )["l2"]
    return dict(
        nfe=nfe,
        naux=naux,
        steps=steps,
        method=method,
        graph=False,
        end=end,
        setup_seconds=setup,
        evolution_seconds=elapsed,
        mass_error=mass_error,
        energy_ratio=energy_ratio,
        physical_energy_bound=physical_bound,
        best_fe_l2=floor,
        **errors
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = validate()
    meta = provenance()
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(dict(result=validation, provenance=meta), indent=2) + "\n"
    )
    with args.output.open("w") as stream:
        for nfe in (32, 64):
            for naux in (32, 64):
                for steps in (1, 4, 16, 64):
                    for method in ("memory", "reset"):
                        case = dict(nfe=nfe, naux=naux, steps=steps, method=method)
                        print(json.dumps(case), flush=True)
                        result = evolve(**case)
                        stream.write(
                            json.dumps(dict(result=result, provenance=meta)) + "\n"
                        )
                        stream.flush()
                        print(json.dumps(result), flush=True)
                        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
