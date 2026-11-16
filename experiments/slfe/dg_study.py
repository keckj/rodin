"""GPU DG/RK4 validation and matched-mesh accuracy/cost measurements.

Distributed under the Boost Software License, Version 1.0.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

from gpu import cp, Mesh, Space, gaussian, quadrature, launch
from eulerian_dg import EulerianDG
from gpu_graph import EvolutionGraph
from day_study import exact_field
from campaign import provenance


def validate(p=2, flux="central"):
    space = Space(Mesh(4), p)
    flow = EulerianDG(space, flux=flux)
    nv = space.mesh.cells * space.nb
    eye = cp.ascontiguousarray(
        cp.eye(nv).reshape(space.mesh.cells, space.nb, nv).transpose(0, 2, 1)
    )
    matrix = flow.rhs(eye).transpose(0, 2, 1).reshape(nv, nv)
    mf = cp.repeat(space.mesh.areas, space.nb)
    normalized = cp.sqrt(mf)[:, None] * matrix / cp.sqrt(mf)[None, :]
    scale = float(cp.max(cp.abs(normalized)))
    skew = float(cp.max(cp.abs(normalized + normalized.T)))
    dissipation = float(cp.linalg.eigvalsh((normalized + normalized.T) / 2)[-1])
    assert dissipation < 1e-11 * max(1.0, scale), dissipation
    if flux == "central":
        assert skew < 1e-12 * max(1.0, scale), skew
    norm = float(cp.linalg.svd(normalized, compute_uv=False)[0])
    assert norm <= flow.norm_bound * (1 + 1e-12)
    one = cp.zeros((space.mesh.cells, 1, space.nb))
    one[:, :, 0] = 1
    constant = float(cp.max(cp.abs(flow.rhs(one))))
    mass = float(cp.max(cp.abs(space.mass(flow.rhs(eye)))))
    assert constant < 2e-10 and mass < 2e-12, (constant, mass)
    result = flow.step(eye, 2 / flow.norm_bound).transpose(0, 2, 1).reshape(nv, nv)
    singular = float(
        cp.linalg.svd(
            cp.sqrt(mf)[:, None] * result / cp.sqrt(mf)[None, :], compute_uv=False
        )[0]
    )
    # RK4 is contractive on the skew-adjoint central operator at this bound.
    # A dissipative nonnormal operator does not inherit that RK4 theorem.
    # Record its actual amplification rather than assuming contractivity.
    if flux == "central":
        assert singular < 1 + 2e-11, singular
    refined = EulerianDG(space, degree=20, face_order=16, flux=flux)
    qerror = max(
        float(cp.max(cp.abs(flow.diagonal - refined.diagonal))),
        float(cp.max(cp.abs(flow.offdiagonal - refined.offdiagonal))),
    )
    if flux == "central":
        assert qerror < 1e-8, qerror
    q = quadrature(16)
    r, s = q[:, 0], q[:, 1]
    basis = space.basis(q[:, :2])
    coefficients = cp.einsum("q,qi,q->i", r**p + s**p, basis, 2 * q[:, 2])
    gradients = cp.empty((len(q), space.nb, 2))
    launch(
        flow.module.get_function("evaluate_gradients"),
        len(q),
        (cp.ascontiguousarray(q[:, :2]), gradients, len(q)),
    )
    exact = cp.stack((p * r ** (p - 1), p * s ** (p - 1)), axis=-1)
    derivative_error = float(
        cp.max(cp.abs(cp.einsum("qia,i->qa", gradients, coefficients) - exact))
    )
    assert derivative_error < 3e-11, derivative_error
    return dict(
        p=p,
        flux=flux,
        skew_error=skew,
        dissipation_max=dissipation,
        constant_error=constant,
        mass_error=mass,
        generator_norm=norm,
        norm_bound=flow.norm_bound,
        rk4_norm=singular,
        rk4_contractive=singular < 1 + 2e-11,
        quadrature_difference=qerror,
        skew_correction=flow.skew_correction,
        derivative_error=derivative_error,
    )


def evolve(
    n=16,
    p=3,
    multiplier=1,
    end=0.5,
    reversing=True,
    graph=True,
    flux="central",
    cfl_scale=1.0,
):
    space = Space(Mesh(n), p)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    flow = EulerianDG(space, flux=flux)
    initial = space.project(gaussian)
    duration = float(cp.sin(cp.asarray(cp.pi * end)) / cp.pi) if reversing else end
    steps = (
        max(1, math.ceil(abs(duration) * flow.norm_bound / (2 * cfl_scale)))
        * multiplier
    )
    dt = duration / steps
    replay = EvolutionGraph(flow.step, initial, dt) if graph else None
    flow.step(initial, dt)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    value = initial
    if graph:
        for _ in range(steps):
            replay.launch(dt)
        value = replay.result()
    else:
        for _ in range(steps):
            value = flow.step(value, dt)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    errors = space.errors(
        value, lambda points, t: exact_field(points, t, reversing), end
    )
    mass = float(cp.max(cp.abs(space.mass(value) - space.mass(initial))))
    energy = float(cp.max(space.energy(value) / space.energy(initial)))
    assert mass < 2e-10 and energy < 1 + 2e-10, (mass, energy)
    return dict(
        n=n,
        p=p,
        steps=steps,
        multiplier=multiplier,
        end=end,
        reversing=reversing,
        graph=graph,
        flux=flux,
        cfl_scale=cfl_scale,
        static_setup_seconds=setup,
        evolution_seconds=elapsed,
        mass_error=mass,
        energy_ratio=energy,
        skew_correction=flow.skew_correction,
        norm_bound=flow.norm_bound,
        storage_bytes=flow.diagonal.nbytes
        + flow.offdiagonal.nbytes
        + flow.neighbor.nbytes,
        **errors
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=(
            "validation",
            "accuracy",
            "upwind-validation",
            "upwind-operator",
            "upwind",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = (
        [("validate", dict(p=p)) for p in (1, 2, 3, 4)]
        if args.suite == "validation"
        else [
            ("evolve", dict(n=n, p=p, multiplier=multiplier))
            for p in (1, 2, 3, 4)
            for n in (8, 16, 32, 64)
            for multiplier in (1, 2)
        ]
    )
    if args.suite in ("upwind-validation", "upwind-operator"):
        jobs = [("validate", dict(p=p, flux="upwind")) for p in (1, 2, 3, 4)]
    if args.suite == "upwind":
        jobs = [
            ("evolve", dict(n=n, p=p, multiplier=multiplier, flux="upwind"))
            for p in (1, 2, 3, 4)
            for n in (8, 16, 32, 64)
            for multiplier in (1, 2)
        ]
    path = args.output / ("dg-" + args.suite + ".jsonl")
    done = (
        {json.loads(l)["case_id"] for l in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    meta = provenance()
    for kind, parameters in jobs:
        key = hashlib.sha256(
            json.dumps([kind, parameters], sort_keys=True).encode()
        ).hexdigest()[:16]
        if key in done:
            continue
        print(json.dumps(dict(start=key, kind=kind, **parameters)), flush=True)
        try:
            record = dict(status="ok", result=globals()[kind](**parameters))
        except Exception:
            record = dict(status="failed", exception=traceback.format_exc())
        record.update(case_id=key, parameters=parameters, provenance=meta)
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(dict(done=key, status=record["status"])), flush=True)
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
