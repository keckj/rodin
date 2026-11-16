"""Validation and accuracy studies for FE/Fourier incremental evolution.

Distributed under the Boost Software License, Version 1.0.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from gpu import cp, Mesh, Space, gaussian
from fourier_carrier import FourierCarrier, UnitaryCellular
from day_study import exact_field
from campaign import provenance
from gpu_graph import EvolutionGraph


def validate(p=2, factor=1):
    space = Space(Mesh(4), p)
    c = FourierCarrier(space, factor=factor)
    refined = FourierCarrier(space, factor=factor, quadrature_order=48)
    moment_error = float(cp.max(cp.abs(c.moments - refined.moments)))
    assert moment_error < 2e-13
    nv = space.mesh.cells * space.nb
    eye = cp.ascontiguousarray(
        cp.eye(nv).reshape(space.mesh.cells, space.nb, nv).transpose(0, 2, 1)
    )
    mf = cp.repeat(space.mesh.areas, space.nb)
    matrix = c.project(eye).reshape(-1, nv)
    singular = cp.linalg.svd(matrix / (c.n * cp.sqrt(mf)[None, :]), compute_uv=False)
    assert float(singular[0]) <= 1 + 2e-11
    random = cp.random.RandomState(418).normal(size=(c.n, c.n, 1))
    oracle = matrix.T @ random.ravel() / (c.n * c.n * mf)
    adjoint_error = float(cp.max(cp.abs(c.adjoint(random).ravel() - oracle)))
    assert adjoint_error < 2e-11
    constant = cp.zeros((space.mesh.cells, 1, space.nb))
    constant[:, :, 0] = 1
    assert float(cp.max(cp.abs(c.project(constant) - 1))) < 2e-11
    assert float(cp.max(cp.abs(c.adjoint(cp.ones_like(random)) - constant))) < 2e-11
    flow = UnitaryCellular(c.n)
    spectra = []
    projected = c.project(eye)
    for duration in (0.0, 1e-6, 0.1, 0.5, 2.0, -0.1):
        transported = flow.step(projected, duration)
        result = eye + c.adjoint(transported - projected)
        transfer = result.transpose(0, 2, 1).reshape(nv, nv)
        norm = float(
            cp.linalg.svd(
                cp.sqrt(mf)[:, None] * transfer / cp.sqrt(mf)[None, :], compute_uv=False
            )[0]
        )
        mass = float(cp.max(cp.abs(space.mass(result) - space.mass(eye))))
        reverse = float(cp.max(cp.abs(flow.step(transported, -duration) - projected)))
        energy = float(
            cp.max(
                cp.abs(
                    cp.sum(transported**2, axis=(0, 1))
                    - cp.sum(projected**2, axis=(0, 1))
                )
            )
        )
        energy /= max(1.0, float(cp.max(cp.sum(projected**2, axis=(0, 1)))))
        assert norm < 1 + 3e-11 and mass < 2e-11 and reverse < 2e-11 and energy < 2e-11
        if duration == 0:
            assert float(cp.max(cp.abs(transfer - cp.eye(nv)))) < 2e-11
        spectra.append(
            dict(
                duration=duration,
                norm=norm,
                mass_error=mass,
                reversal_error=reverse,
                energy_error=energy,
            )
        )
    return dict(
        p=p,
        factor=factor,
        modes=c.n,
        moment_error=moment_error,
        adjoint_error=adjoint_error,
        projection_norm=float(singular[0]),
        projection_min=float(singular[-1]),
        spectra=spectra,
    )


def evolve(
    n=16,
    p=3,
    factor=1,
    steps=16,
    end=0.5,
    reversing=True,
    order=4,
    persistent=True,
    modes=None,
    graph=False,
):
    space = Space(Mesh(n), p)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    carrier = FourierCarrier(space, factor=factor, modes=modes)
    flow = UnitaryCellular(carrier.n)
    initial = space.project(gaussian)
    x = initial
    projected = carrier.project(initial)
    y = projected
    durations = []
    for k in range(steps):
        t0, t1 = end * k / steps, end * (k + 1) / steps
        durations.append(
            (cp.sin(cp.asarray(cp.pi * t1)) - cp.sin(cp.asarray(cp.pi * t0))) / cp.pi
            if reversing
            else t1 - t0
        )
    flow.step(y, durations[0], order=order)  # Warm FFT plans and kernels.
    replay = (
        EvolutionGraph(lambda v, dt: flow.step(v, dt, order=order), y, durations[0])
        if graph
        else None
    )
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    if persistent:
        # Count both FE exchanges in evolution cost, even though the initial
        # projection is also used above to warm plans.
        projected = carrier.project(initial)
        y = projected
        if graph:
            replay.reset(y)
            for duration in durations:
                replay.launch(duration)
            y = replay.result()
        else:
            for duration in durations:
                y = flow.step(y, duration, order=order)
        x = initial + carrier.adjoint(y - projected)
    else:
        if graph:
            raise ValueError("Graph replay is currently a persistent-state experiment")
        for duration in durations:
            old = carrier.project(x)
            x = x + carrier.adjoint(flow.step(old, duration, order=order) - old)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    result = space.errors(x, lambda points, t: exact_field(points, t, reversing), end)
    mass = float(cp.max(cp.abs(space.mass(x) - space.mass(initial))))
    energy = float(cp.max(space.energy(x) / space.energy(initial)))
    assert mass < 2e-10 and energy <= 1 + 2e-10, (mass, energy)
    return dict(
        n=n,
        p=p,
        factor=factor,
        modes=carrier.n,
        steps=steps,
        end=end,
        reversing=reversing,
        order=order,
        persistent=persistent,
        graph=graph,
        static_setup_seconds=setup,
        evolution_seconds=elapsed,
        mass_error=mass,
        energy_ratio=energy,
        coupling_bytes=carrier.dg.blocks.nbytes,
        **result
    )


def cases(suite):
    if suite == "validation":
        for p in (1, 2, 3, 4):
            for factor in (1, 2):
                yield "validate", dict(p=p, factor=factor)
    elif suite == "accuracy":
        for n in (8, 16, 32, 64):
            for factor in (1, 2):
                for steps in (4, 8, 16, 32, 64):
                    yield "evolve", dict(n=n, factor=factor, steps=steps)
    elif suite == "time-order":
        for persistent in (False, True):
            for order in (2, 4):
                for steps in (4, 8, 16, 32, 64, 128):
                    yield "evolve", dict(
                        n=32, factor=2, steps=steps, order=order, persistent=persistent
                    )
    elif suite == "autonomous":
        for n in (16, 32, 64):
            for steps in (8, 16, 32, 64):
                yield "evolve", dict(n=n, factor=2, steps=steps, reversing=False)
    elif suite == "compressed":
        for p in (3, 4):
            for n in (16, 32, 64):
                for modes in (63, 95, 127):
                    for steps in (8, 16, 32):
                        yield "evolve", dict(
                            n=n, p=p, factor=1, modes=modes, steps=steps, graph=True
                        )
    else:
        raise ValueError(suite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / ("fourier-" + args.suite + ".jsonl")
    done = (
        {json.loads(l)["case_id"] for l in path.read_text().splitlines() if l.strip()}
        if path.exists()
        else set()
    )
    meta = provenance()
    for kind, parameters in cases(args.suite):
        key = hashlib.sha256(
            json.dumps([kind, parameters], sort_keys=True).encode()
        ).hexdigest()[:16]
        if key in done:
            continue
        print(json.dumps(dict(start=key, kind=kind, **parameters)), flush=True)
        start = time.perf_counter()
        try:
            record = dict(status="ok", result=globals()[kind](**parameters))
        except Exception:
            record = dict(status="failed", exception=traceback.format_exc())
        record.update(
            case_id=key,
            parameters=parameters,
            provenance=meta,
            wall_seconds=time.perf_counter() - start,
        )
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(
            json.dumps(
                dict(done=key, status=record["status"], seconds=record["wall_seconds"])
            ),
            flush=True,
        )
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
