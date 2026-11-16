"""Daytime research studies: oriented carriers and general cellular transport.

Distributed under the Boost Software License, Version 1.0.
"""

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from gpu import cp, HERE, Mesh, Space, gaussian, launch
from carrier import Carrier
from fourier_flow import (
    CellularTransport,
    NestedCellularTransport,
    validate_decomposition,
)
from campaign import provenance


@lru_cache(None)
def reference_kernel():
    kernel = cp.RawKernel(
        (HERE / "cellular_reference.cu").read_text(), "reference_feet"
    )
    kernel.compile()
    return kernel


def reference_feet(points, duration, steps=512):
    original = points.shape
    points = cp.ascontiguousarray(points.reshape(-1, 2))
    feet = cp.empty_like(points)
    launch(
        reference_kernel(), len(points), (points, feet, len(points), duration, steps)
    )
    return feet.reshape(original)


def exact_field(points, time, reversing, profile=gaussian, substeps=512):
    duration = float(cp.sin(cp.asarray(cp.pi * time)) / cp.pi) if reversing else time
    if reversing and time == 1.0:
        return profile(points, 0)
    return profile(reference_feet(points, -duration, substeps), 0)


def quadrature_blocks(carrier, velocity, dt):
    # Independent original quadrature construction retained only as an oracle.
    def lagrange(x):
        result = cp.ones((*x.shape, carrier.np), dtype=cp.float64)
        for i in range(carrier.np):
            for j in range(carrier.np):
                if i != j:
                    result[..., i] *= (x - carrier.nodes[j]) / (
                        carrier.nodes[i] - carrier.nodes[j]
                    )
        return result

    displacement = carrier.n * dt * velocity
    shift = cp.floor(displacement).astype(cp.int32)
    fraction = displacement - shift
    points = cp.stack(
        (
            fraction[:, None] * carrier.nodes[None, :],
            fraction[:, None] + (1 - fraction[:, None]) * carrier.nodes[None, :],
        ),
        axis=1,
    )
    origins = points - fraction[:, None, None] + cp.asarray([1.0, 0.0])[None, :, None]
    lengths = cp.stack((fraction, 1 - fraction), axis=1)
    blocks = cp.einsum(
        "rsqi,rsqj,q,rs->rsij",
        lagrange(points),
        lagrange(origins),
        carrier.weights,
        lengths,
    )
    return shift, blocks / carrier.weights[None, None, :, None]


def kernel_validation():
    checks = []
    for p in (0, 1, 2, 3, 4, 5):
        c = Carrier(Space(Mesh(4), min(p, 3)), degree=p)
        fractions = cp.random.RandomState(519).uniform(-100.0, 100.0, c.n * c.np)
        special = cp.asarray([0.0, 1.0, -1.0, 1e-12, -1e-12, 1.0 - 1e-12])
        fractions[: min(len(fractions), len(special))] = special[
            : min(len(fractions), len(special))
        ]
        velocity = fractions / c.n
        actual = c.shears([(0, 1.0, velocity)]).operators[0]
        shift, oracle = quadrature_blocks(c, velocity, 1.0)
        error = float(cp.max(cp.abs(actual[2] - oracle)))
        assert bool(cp.all(actual[1] == shift))
        assert error < 2e-12, (p, error)
        checks.append(dict(degree=p, max_entry_error=error))
    points = cp.random.RandomState(917).uniform(0, 1, (4096, 2))
    reference = []
    for duration in (0.1, 0.5, 1.0):
        a = reference_feet(points, duration, 512)
        b = reference_feet(points, duration, 1024)
        difference = float(cp.max(cp.abs(a - b)))
        assert difference < 3e-10, (duration, difference)
        reference.append(dict(duration=duration, refinement_difference=difference))
    return dict(
        passed=True,
        kernel_checks=checks,
        reference_checks=reference,
        velocity_decomposition_error=validate_decomposition(),
    )


def oriented_validation(p=2, factor=1):
    space = Space(Mesh(4), p)
    flow = CellularTransport(space, factor=factor)
    ndof = space.mesh.cells * space.nb
    eye = cp.ascontiguousarray(
        cp.eye(ndof).reshape(space.mesh.cells, space.nb, ndof).transpose(0, 2, 1)
    )
    mf = cp.repeat(space.mesh.areas, space.nb)
    gram = cp.zeros((ndof, ndof), dtype=cp.float64)
    carrier_norms = []
    for carrier in flow.carriers:
        matrix = carrier.project(eye).transpose(0, 2, 1).reshape(-1, ndof)
        mc = cp.tile(carrier.mass, carrier.n**2)
        weighted = cp.sqrt(mc)[:, None] * matrix / cp.sqrt(mf)[None, :]
        sv = cp.linalg.svd(weighted, compute_uv=False)
        assert float(sv[0]) < 1 + 2e-11
        gram += weighted.T @ weighted / len(flow.carriers)
        carrier_norms.append(float(sv[0]))
    eigenvalues = cp.linalg.eigvalsh(gram)
    spectra = []
    one = cp.zeros((space.mesh.cells, 1, space.nb), dtype=cp.float64)
    one[:, :, 0] = 1
    for dt in (0.0, 1e-6, 0.1, 0.5, -0.1):
        step = flow.step(0, dt, reversible=False)
        matrix = step.apply(eye).transpose(0, 2, 1).reshape(ndof, ndof)
        norm = float(
            cp.linalg.svd(
                cp.sqrt(mf)[:, None] * matrix / cp.sqrt(mf)[None, :], compute_uv=False
            )[0]
        )
        mass = float(cp.max(cp.abs(space.mass(step.apply(eye)) - space.mass(eye))))
        constant = float(cp.max(cp.abs(step.apply(one) - one)))
        assert norm < 1 + 3e-11 and mass < 1e-11 and constant < 1e-11
        if dt == 0:
            assert float(cp.max(cp.abs(matrix - cp.eye(ndof)))) < 1e-12
        spectra.append(dict(dt=dt, norm=norm, mass_error=mass, constant_error=constant))
    return dict(
        passed=True,
        p=p,
        factor=factor,
        carrier_norms=carrier_norms,
        combined_gram_min=float(eigenvalues[0]),
        combined_nullity=int(cp.sum(eigenvalues < 1e-12)),
        spectra=spectra,
    )


def nested_validation(p=2, factor=1):
    space = Space(Mesh(4), p)
    flow = NestedCellularTransport(space, factor=factor)
    c = flow.carrier
    nw = c.n * c.n * c.nc
    wc = cp.ascontiguousarray(
        cp.eye(nw).reshape(c.n * c.n, c.nc, nw).transpose(0, 2, 1)
    )
    mc = cp.tile(c.mass, c.n * c.n)
    forward = flow.plus.apply_carrier(wc).transpose(0, 2, 1).reshape(nw, nw)
    backward = flow.minus.apply_carrier(wc).transpose(0, 2, 1).reshape(nw, nw)
    adjoint_error = float(
        cp.max(cp.abs(mc[:, None] * forward - backward.T * mc[None, :]))
    )
    assert adjoint_error < 2e-13, adjoint_error
    nv = space.mesh.cells * space.nb
    eye = cp.ascontiguousarray(
        cp.eye(nv).reshape(space.mesh.cells, space.nb, nv).transpose(0, 2, 1)
    )
    mf = cp.repeat(space.mesh.areas, space.nb)
    one = cp.zeros((space.mesh.cells, 1, space.nb))
    one[:, :, 0] = 1
    spectra = []
    for dt in (0.0, 1e-6, 0.1, 0.5, 2.0, -0.1):
        step = flow.step(0, dt, reversible=False)
        matrix = step.apply(eye).transpose(0, 2, 1).reshape(nv, nv)
        norm = float(
            cp.linalg.svd(
                cp.sqrt(mf)[:, None] * matrix / cp.sqrt(mf)[None, :], compute_uv=False
            )[0]
        )
        mass = float(cp.max(cp.abs(space.mass(step.apply(eye)) - space.mass(eye))))
        constant = float(cp.max(cp.abs(step.apply(one) - one)))
        assert norm < 1 + 3e-11 and mass < 1e-11 and constant < 1e-11, (
            norm,
            mass,
            constant,
        )
        if dt == 0:
            assert float(cp.max(cp.abs(matrix - cp.eye(nv)))) < 1e-12
        spectra.append(dict(dt=dt, norm=norm, mass_error=mass, constant_error=constant))
    return dict(
        passed=True,
        p=p,
        factor=factor,
        coordinate_adjoint_error=adjoint_error,
        spectra=spectra,
    )


def evolve(
    n=16,
    p=3,
    steps=64,
    end=1.0,
    factor=1,
    degree=None,
    reversing=True,
    grouped=True,
    seed=17,
    method="oriented",
    persistent=False,
):
    space = Space(Mesh(n, seed=seed), p)
    constructor = {"oriented": CellularTransport, "nested": NestedCellularTransport}[
        method
    ]
    flow = constructor(space, factor=factor, degree=degree, grouped=grouped)
    x = space.project(gaussian)
    mass0 = space.mass(x)
    energy0 = space.energy(x)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    if persistent:
        if method != "nested":
            raise ValueError("Persistent evolution requires one carrier")
        initial = flow.carrier.project(x)
        auxiliary = initial
        for k in range(steps):
            auxiliary = flow.step(
                end * k / steps, end * (k + 1) / steps, reversible=reversing
            ).apply_carrier(auxiliary)
        x = x + flow.carrier.adjoint(auxiliary - initial)
    else:
        for k in range(steps):
            x = flow.step(
                end * k / steps, end * (k + 1) / steps, reversible=reversing
            ).apply(x)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    result = space.errors(x, lambda points, t: exact_field(points, t, reversing), end)
    mass = float(cp.max(cp.abs(space.mass(x) - mass0)))
    energy = float(cp.max(space.energy(x) / energy0))
    assert mass < 1e-9 and energy < 1 + 1e-9, (mass, energy)
    return dict(
        n=n,
        p=p,
        steps=steps,
        end=end,
        factor=factor,
        degree=degree,
        reversing=reversing,
        grouped=grouped,
        seed=seed,
        method=method,
        persistent=persistent,
        static_setup_seconds=flow.build_seconds,
        evolution_seconds=elapsed,
        storage_bytes=flow.storage_bytes,
        mass_error=mass,
        energy_ratio=energy,
        **result
    )


def cases(suite):
    if suite == "validation":
        yield "kernel_validation", {}
        for p in (1, 2, 3):
            for factor in (1, 2):
                yield "oriented_validation", dict(p=p, factor=factor)
    elif suite == "cellular":
        for end in (0.5, 1.0):
            for p in (1, 2, 3):
                for n in (8, 16, 32, 64):
                    for factor in (1, 2):
                        yield "evolve", dict(n=n, p=p, steps=64, end=end, factor=factor)
    elif suite == "temporal":
        for reversing in (True, False):
            for n in (16, 32, 64):
                for steps in (8, 16, 32, 64, 128, 256, 512):
                    yield "evolve", dict(
                        n=n, p=3, steps=steps, end=0.5, factor=2, reversing=reversing
                    )
    elif suite == "enrichment":
        for n in (8, 16, 32, 64):
            for degree, factor in ((3, 1), (3, 2), (4, 1), (5, 1)):
                yield "evolve", dict(
                    n=n, p=3, steps=64, end=0.5, factor=factor, degree=degree
                )
    elif suite == "seeds":
        for seed in range(1, 31):
            for p in (2, 3):
                for factor in (1, 2):
                    yield "evolve", dict(
                        n=16, p=p, steps=64, end=0.5, factor=factor, seed=seed
                    )
    elif suite == "nested":
        for n in (8, 16, 32, 64):
            for factor in (1, 2):
                for end in (0.5, 1.0):
                    for steps in (64, 256):
                        yield "evolve", dict(
                            n=n,
                            p=3,
                            steps=steps,
                            end=end,
                            factor=factor,
                            method="nested",
                        )
    elif suite == "nested-validation":
        for p in (1, 2, 3):
            for factor in (1, 2):
                yield "nested_validation", dict(p=p, factor=factor)
    elif suite == "persistent":
        for n in (8, 16, 32, 64):
            for factor in (1, 2):
                for end in (0.5, 1.0):
                    for steps in (64, 256):
                        yield "evolve", dict(
                            n=n,
                            p=3,
                            steps=steps,
                            end=end,
                            factor=factor,
                            method="nested",
                            persistent=True,
                        )
    else:
        raise ValueError(suite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / ("day-" + args.suite + ".jsonl")
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
