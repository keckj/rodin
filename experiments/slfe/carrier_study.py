"""Reproducible GPU validation of incremental auxiliary-space transport.

Distributed under the Boost Software License, Version 1.0.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from gpu import cp, Mesh, Space, smooth, timings
from carrier import Carrier
from flows import shear_solution
from campaign import provenance


def validate(p, factor):
    mesh = Mesh(4)
    space = Space(mesh, p)
    carrier = Carrier(space, 4*factor)
    ndof = mesh.cells*space.nb
    eye = cp.ascontiguousarray(cp.eye(ndof).reshape(mesh.cells, space.nb, ndof).transpose(0, 2, 1))
    projected = carrier.project(eye)
    matrix = projected.transpose(0, 2, 1).reshape(-1, ndof)
    mf = cp.repeat(mesh.areas, space.nb)
    mc = cp.tile(carrier.mass, carrier.n**2)
    normalized = cp.sqrt(mc)[:, None]*matrix/cp.sqrt(mf)[None, :]
    sv = cp.linalg.svd(normalized, compute_uv=False)
    assert float(sv[0]) <= 1+1e-11
    one = cp.zeros((mesh.cells, 1, space.nb), dtype=cp.float64)
    one[:, :, 0] = 1
    assert float(cp.max(cp.abs(carrier.project(one)-1))) < 1e-11
    reverse = carrier.adjoint(cp.ones((carrier.n**2, 1, carrier.nc)))
    assert float(cp.max(cp.abs(reverse-one))) < 1e-11
    # The independent transpose formula uses the dense cross-mass action.
    random = cp.random.RandomState(314).normal(size=(carrier.n**2, 1, carrier.nc))
    adjoint = carrier.adjoint(random).ravel()
    oracle = matrix.T@(mc*random.ravel())/mf
    adjoint_error = float(cp.max(cp.abs(adjoint-oracle)))
    assert adjoint_error < 1e-11
    for k in range(2*carrier.np):
        assert abs(float(cp.sum(carrier.weights*carrier.nodes**k))-1/(k+1)) < 1e-13
    other = Carrier(space, 4*factor, qdegree=3*p+2)
    quadrature_error = float(cp.max(cp.abs(other.project(eye)-projected)))
    assert quadrature_error < 1e-10
    spectra = []
    for flow in ("translation", "shear"):
        for dt in (0., 1e-6, .037, .71):
            transfer = carrier.transport(dt, flow)
            result = transfer.apply(eye).transpose(0, 2, 1).reshape(ndof, ndof)
            weighted = cp.sqrt(mf)[:, None]*result/cp.sqrt(mf)[None, :]
            singular = cp.linalg.svd(weighted, compute_uv=False)
            constant_error = float(cp.max(cp.abs(transfer.apply(one)-one)))
            mass_error = float(cp.max(cp.abs(space.mass(transfer.apply(eye))-space.mass(eye))))
            identity_error = float(cp.max(cp.abs(result-cp.eye(ndof)))) if dt == 0 else None
            assert float(singular[0]) <= 1+2e-11
            assert constant_error < 1e-11 and mass_error < 1e-11
            if dt == 0:
                assert identity_error < 1e-12
            spectra.append(dict(flow=flow, dt=dt, singular_max=float(singular[0]),
                                constant_error=constant_error, mass_error=mass_error, identity_error=identity_error))
    return dict(p=p, factor=factor, passed=True, projection_singular_max=float(sv[0]),
                projection_singular_min=float(sv[-1]), projection_nullity=ndof-len(sv),
                adjoint_error=adjoint_error, quadrature_error=quadrature_error, spectra=spectra)


def evolve(n, p, steps=16, end=1., flow="shear", factor=1, incremental=True, seed=17):
    mesh = Mesh(n, seed=seed)
    space = Space(mesh, p)
    carrier = Carrier(space, n*factor)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    transfer = carrier.transport(end/steps, flow)
    cp.cuda.Stream.null.synchronize()
    kernel_setup = time.perf_counter()-start
    exact = smooth if flow == "translation" else shear_solution
    x = space.project(smooth)
    mass0, energy0 = space.mass(x), space.energy(x)
    application_ms = timings(lambda: transfer.apply(x, incremental), repeats=12)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        x = transfer.apply(x, incremental)
    cp.cuda.Stream.null.synchronize()
    wall = time.perf_counter()-start
    return dict(n=n, p=p, steps=steps, end=end, flow=flow, factor=factor, seed=seed,
                incremental=incremental, static_setup_seconds=carrier.build_seconds,
                transport_setup_seconds=kernel_setup, application_ms=application_ms,
                evolution_seconds=wall, coupling_bytes=carrier.blocks.nbytes,
                mass_error=float(cp.max(cp.abs(space.mass(x)-mass0))),
                energy_ratio=float(cp.max(space.energy(x)/energy0)), **space.errors(x, exact, end))


def cases(suite):
    if suite == "validation":
        for p in (1, 2, 3):
            for factor in (1, 2):
                yield "validate", dict(p=p, factor=factor)
    elif suite == "convergence":
        for flow in ("translation", "shear"):
            for p in (1, 2, 3):
                for n in (8, 16, 32, 64, 128):
                    yield "evolve", dict(n=n, p=p, flow=flow)
    elif suite == "small_steps":
        for p in (1, 2, 3):
            for incremental in (True, False):
                for steps in (16, 160, 1600, 16000):
                    yield "evolve", dict(n=16, p=p, steps=steps, end=.1,
                                         flow="translation", incremental=incremental)
    elif suite == "long_time":
        for p in (2, 3):
            for factor in (1, 2):
                yield "evolve", dict(n=32, p=p, steps=10000, end=10., flow="translation", factor=factor)
    elif suite == "resolution":
        for factor in (1, 2, 3):
            for n in (16, 32, 64):
                yield "evolve", dict(n=n, p=3, factor=factor)
    else:
        raise ValueError(suite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    meta = provenance()
    path = args.output/("carrier-"+args.suite+".jsonl")
    done = {json.loads(x)["case_id"] for x in path.read_text().splitlines() if x.strip()} if path.exists() else set()
    for kind, parameters in cases(args.suite):
        key = hashlib.sha256(json.dumps([kind, parameters], sort_keys=True).encode()).hexdigest()[:16]
        if key in done:
            continue
        start = time.perf_counter()
        print(json.dumps(dict(start=key, **parameters)), flush=True)
        try:
            result = globals()[kind](**parameters)
            record = dict(status="ok", result=result)
        except Exception:
            record = dict(status="failed", exception=traceback.format_exc())
        record.update(case_id=key, parameters=parameters, provenance=meta, wall_seconds=time.perf_counter()-start)
        with path.open("a") as stream:
            stream.write(json.dumps(record)+"\n"); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(dict(done=key, status=record["status"], seconds=record["wall_seconds"])), flush=True)
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
