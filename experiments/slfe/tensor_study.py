"""Validate and measure locally resolved tensor carriers, using CUDA only.

Distributed under the Boost Software License, Version 1.0.
Translation and horizontal shear isolate auxiliary resolution from the
additional coordinate projections required by a cellular flow.
"""

import argparse
import json
import math
from pathlib import Path
import time

from gpu import cp, Space
from graded_study import GradedMesh
from carrier import Carrier
from campaign import provenance


def partitions(n, grading):
    t = cp.linspace(0.0, 1.0, n + 1)
    return cp.stack(
        [
            t + grading / (2 * cp.pi) * cp.sin(2 * cp.pi * (t + offset))
            for offset in (0.15, 0.05)
        ]
    )


def validate(p=2, grading=0.9):
    s = Space(GradedMesh(4, 0.6), p)
    c = Carrier(s, edges=partitions(4, grading))
    nd = s.mesh.cells * s.nb
    eye = cp.ascontiguousarray(
        cp.eye(nd).reshape(s.mesh.cells, s.nb, nd).transpose(0, 2, 1)
    )
    projected = c.project(eye)
    matrix = projected.transpose(0, 2, 1).reshape(-1, nd)
    mf = cp.repeat(s.mesh.areas, s.nb)
    mc = c.mass.ravel()
    weighted = cp.sqrt(mc)[:, None] * matrix / cp.sqrt(mf)[None, :]
    singular = float(cp.linalg.svd(weighted, compute_uv=False)[0])
    assert singular < 1 + 2e-11, singular
    one = cp.zeros((s.mesh.cells, 1, s.nb))
    one[:, :, 0] = 1
    constant = float(cp.max(cp.abs(c.project(one) - 1)))
    constant_back = float(cp.max(cp.abs(c.adjoint(cp.ones((c.n**2, 1, c.nc))) - one)))
    random = cp.random.RandomState(314).normal(size=(c.n**2, 1, c.nc))
    adjoint = float(
        cp.max(
            cp.abs(c.adjoint(random).ravel() - matrix.T @ (mc * random.ravel()) / mf)
        )
    )
    assert max(constant, constant_back, adjoint) < 3e-11
    other = Carrier(
        s, edges=partitions(4, grading), qdegree=min(20, 3 * p + 2), builder="scalar"
    )
    quadrature = float(cp.max(cp.abs(other.project(eye) - projected)))
    assert quadrature < 3e-11, quadrature
    records = []
    # Dense carrier-space SVD independently checks the remap itself.
    nw = c.n**2 * c.nc
    identity = cp.ascontiguousarray(
        cp.eye(nw).reshape(c.n**2, c.nc, nw).transpose(0, 2, 1)
    )
    for flow in ("translation", "shear"):
        for dt in (0.0, 1e-6, 0.037, 0.71, -0.71, 2.31):
            transfer = c.transport(dt, flow)
            result = transfer.apply_carrier(identity).transpose(0, 2, 1).reshape(nw, nw)
            normalized = cp.sqrt(mc)[:, None] * result / cp.sqrt(mc)[None, :]
            sv = float(cp.linalg.svd(normalized, compute_uv=False)[0])
            constant_error = float(cp.max(cp.abs(result.sum(axis=1) - 1)))
            mass_error = float(cp.max(cp.abs(mc @ result - mc)))
            assert sv < 1 + 3e-11, sv
            assert max(constant_error, mass_error) < 3e-11
            if dt == 0:
                assert float(cp.max(cp.abs(result - cp.eye(nw)))) < 3e-11
            fe = transfer.apply(eye).transpose(0, 2, 1).reshape(nd, nd)
            fesv = float(
                cp.linalg.svd(
                    cp.sqrt(mf)[:, None] * fe / cp.sqrt(mf)[None, :], compute_uv=False
                )[0]
            )
            assert fesv < 1 + 4e-11
            records.append(
                dict(
                    flow=flow,
                    dt=dt,
                    singular_max=sv,
                    fe_singular_max=fesv,
                    constant_error=constant_error,
                    mass_error=mass_error,
                )
            )
    return dict(
        p=p,
        grading=grading,
        projection_singular_max=singular,
        constant_error=constant,
        adjoint_constant_error=constant_back,
        adjoint_error=adjoint,
        quadrature_error=quadrature,
        remaps=records,
    )


def evolve(n=64, grading=0.95, steps=1, flow="translation", with_fields=False):
    s = Space(GradedMesh(64, 0.95), 4)
    width = 0.0015
    end = 0.005

    def profile(points, t):
        delta = cp.sin(cp.pi * (points - cp.asarray([0.35, 0.45])))
        return cp.exp(-cp.sum(delta * delta, axis=-1) / (2 * (cp.pi * width) ** 2)) / (
            math.sqrt(math.pi) * width
        )

    def exact(points, t):
        feet = points.copy()
        if flow == "translation":
            feet -= end * cp.asarray([1.0, 0.37])
        elif flow == "cellular":
            from day_study import reference_feet

            feet = reference_feet(points, -end, 512)
        else:
            feet[..., 0] -= end * 0.2 * cp.sin(2 * cp.pi * points[..., 1])
        return profile(feet, 0)

    initial = s.project(profile, degree=20)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    c = Carrier(s, n=n, edges=partitions(n, grading) if grading else None)
    if flow == "cellular":
        from tensor_cellular import TensorCellular

        transport = TensorCellular(c, end / steps, grading=grading)
    else:
        transport = c.transport(end / steps, flow)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    old = c.project(initial)
    w = old
    for _ in range(steps):
        w = transport.apply_carrier(w)
    result = initial + c.adjoint(w - old)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    mass = float(cp.max(cp.abs(s.mass(result) - s.mass(initial))))
    energy = float(cp.max(s.energy(result) / s.energy(initial)))
    assert mass < 3e-10 and energy < 1 + 3e-10, (mass, energy)
    captured = float(
        cp.sum(old * old * (c.mass[:, None, :] if grading else c.mass))
        / s.energy(initial)[0]
    )
    errors = s.errors(result, exact, 0, degree=20, subdivisions=2)
    best = s.errors(s.project(exact, degree=20), exact, 0, degree=20, subdivisions=2)[
        "l2"
    ]
    record = dict(
        n=n,
        grading=grading,
        steps=steps,
        flow=flow,
        end=end,
        auxiliary_dofs=n * n * c.nc,
        captured_energy=captured,
        setup_seconds=setup,
        evolution_seconds=elapsed,
        coupling_bytes=c.blocks.nbytes,
        mass_error=mass,
        energy_ratio=energy,
        best_fe_l2=best,
        **errors
    )
    return (s, result, initial, exact, record) if with_fields else record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=("validation", "resolution", "cellular"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = provenance()
    if args.suite == "validation":
        cases = [(validate, dict(p=p)) for p in (1, 2, 4)]
    elif args.suite == "cellular":
        cases = [
            (evolve, dict(n=n, grading=0.95, steps=steps, flow="cellular"))
            for n in (64, 128)
            for steps in (1, 4, 16, 64)
        ]
    else:
        cases = [
            (evolve, dict(n=n, grading=g, steps=steps, flow=flow))
            for flow in ("translation", "shear")
            for n, g in ((64, 0.0), (64, 0.95), (128, 0.95))
            for steps in (1, 16)
        ]
    with args.output.open("w") as stream:
        for function, parameters in cases:
            print(json.dumps(parameters), flush=True)
            result = function(**parameters)
            stream.write(
                json.dumps(dict(parameters=parameters, result=result, provenance=meta))
                + "\n"
            )
            stream.flush()
            print(json.dumps(result), flush=True)
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
