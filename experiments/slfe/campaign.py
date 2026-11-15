"""Reproducible GPU campaign with per-case durable records and failure isolation.

Distributed under the Boost Software License, Version 1.0.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

from gpu import cp, HERE, Mesh, Space, Transfer, translation, shear, smooth, gaussian, constant, timings
from cartesian import CartesianTransfer, coordinates


def provenance():
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in HERE.iterdir() if p.suffix in (".py", ".cu", ".cuh", ".json")},
            "python": platform.python_version(), "cupy": cp.__version__,
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion()}


def translation_case(n, p, steps, mode="geometric", qdegree=None, subdivisions=1,
                     end=1., profile="smooth", seed=17, jitter=.18, fields=1):
    dt = end/steps
    mesh = Mesh(n, jitter, seed)
    space = Space(mesh, p)
    function = smooth if profile == "smooth" else gaussian
    coefficients = cp.repeat(space.project(function), fields, axis=1)
    initial_mass = space.mass(coefficients)
    initial_energy = space.energy(coefficients)
    transfer = Transfer(space, mesh.mapped(translation, 0, dt), mode=mode,
                        qdegree=qdegree, subdivisions=subdivisions)
    one = cp.zeros((mesh.cells, 1, space.nb), dtype=cp.float64)
    one[:, :, 0] = 1.
    constant_defect = float(cp.max(cp.abs(transfer.apply(one)-one)))
    scratch = cp.empty_like(coefficients)
    application_ms = timings(lambda: transfer.apply(coefficients, scratch), repeats=15)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    completed = 0
    for k in range(steps):
        transfer.apply(coefficients, scratch)
        coefficients, scratch = scratch, coefficients
        completed = k+1
        if completed % 20 == 0:
            energy = float(cp.max(space.energy(coefficients)/initial_energy))
            if not (energy < 1e12):
                break
    cp.cuda.Stream.null.synchronize()
    evolution_seconds = time.perf_counter()-start
    record = {"n": n, "p": p, "steps": steps, "completed": completed, "end": end,
              "dt": dt, "courant_x": dt*n, "mode": mode, "profile": profile,
              "seed": seed, "jitter": jitter, "fields": fields, "cells": mesh.cells,
              "dofs_per_field": mesh.cells*space.nb, "pairs": transfer.pairs,
              "quadrature_per_cell": transfer.nq, "subdivisions": subdivisions,
              "build_seconds": transfer.build_seconds, "application_ms": application_ms,
              "evolution_seconds": evolution_seconds, "storage_bytes": transfer.storage_bytes,
              "constant_defect": constant_defect,
              "mass_absolute_error": cp.abs(space.mass(coefficients)-initial_mass).get().tolist(),
              "energy_ratio": (space.energy(coefficients)/initial_energy).get().tolist(),
              **space.errors(coefficients, function, completed*dt)}
    return record


def cartesian_case(n, steps, kernel, end=1., profile="smooth", fields=1):
    dt = end/steps
    x = coordinates(n)
    function = smooth if profile == "smooth" else gaussian
    initial = cp.repeat(function(x, 0)[..., None], fields, axis=-1)
    coefficients = initial.copy()
    transfer = CartesianTransfer(n, dt, kernel)
    output, temporary = cp.empty_like(coefficients), cp.empty_like(coefficients)
    application_ms = timings(lambda: transfer.apply(coefficients, output, temporary), repeats=15)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        transfer.apply(coefficients, output, temporary)
        coefficients, output = output, coefficients
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter()-start
    exact = function(x, end)[..., None]
    error = coefficients-exact
    return {"n": n, "steps": steps, "kernel": kernel, "end": end,
            "profile": profile, "fields": fields, "dofs_per_field": n*n,
            "application_ms": application_ms, "evolution_seconds": elapsed,
            "l1": cp.mean(cp.abs(error), axis=(0, 1)).get().tolist(),
            "l2": cp.sqrt(cp.mean(error**2, axis=(0, 1))).get().tolist(),
            "linf_sampled": cp.max(cp.abs(error), axis=(0, 1)).get().tolist(),
            "mass_absolute_error": cp.abs(cp.mean(coefficients-initial, axis=(0, 1))).get().tolist(),
            "min_sampled": cp.min(coefficients, axis=(0, 1)).get().tolist(),
            "max_sampled": cp.max(coefficients, axis=(0, 1)).get().tolist(),
            "error_measure": "uniform nodal discrete norm, not FE integrated error"}


def spectrum_case(n, p, mode, qdegree=None, subdivisions=1, dt=.037, seed=17):
    mesh = Mesh(n, seed=seed)
    space = Space(mesh, p)
    transfer = Transfer(space, mesh.mapped(translation, 0, dt), mode=mode,
                        qdegree=qdegree, subdivisions=subdivisions)
    nb = space.nb
    matrix = cp.zeros((mesh.cells*nb, mesh.cells*nb), dtype=cp.float64)
    rows = transfer.destinations[:, None, None]*nb+cp.arange(nb)[None, :, None]
    cols = transfer.sources[:, None, None]*nb+cp.arange(nb)[None, None, :]
    cp.add.at(matrix, (cp.broadcast_to(rows, transfer.blocks.shape).ravel(),
                       cp.broadcast_to(cols, transfer.blocks.shape).ravel()), transfer.blocks.ravel())
    root = cp.repeat(cp.sqrt(mesh.areas), nb)
    normalized = root[:, None]*matrix/root[None, :]
    singular = cp.linalg.svd(normalized, compute_uv=False)
    return {"n": n, "p": p, "mode": mode, "qdegree": qdegree,
            "subdivisions": subdivisions, "dt": dt, "seed": seed,
            "singular_max": float(singular[0]), "singular_min": float(singular[-1]),
            "singular_above_one": int(cp.sum(singular > 1+1e-10)),
            "largest_ten": singular[:10].get().tolist()}


def cases(suite):
    if suite == "translation":
        for p in [0, 1, 2, 3]:
            for n in [8, 16, 32, 64, 128]:
                for steps in [4, 16, 64, 256]:
                    yield "translation", dict(n=n, p=p, steps=steps)
        for n in [16, 32, 64]:
            for p in [1, 2]:
                for degree, subdivisions in [(4, 1), (8, 1), (12, 1), (12, 2), (12, 4)]:
                    for steps in [4, 32, 128]:
                        yield "translation", dict(n=n, p=p, steps=steps, mode="quadrature",
                                                  qdegree=degree, subdivisions=subdivisions)
    elif suite == "spectrum":
        for p in [0, 1, 2, 3]:
            yield "spectrum", dict(n=8, p=p, mode="geometric")
            for qdegree, subdivisions in [(max(1, 2*p), 1), (12, 1), (12, 2), (12, 4)]:
                yield "spectrum", dict(n=8, p=p, mode="quadrature", qdegree=qdegree, subdivisions=subdivisions)
    elif suite == "cartesian":
        for n in [16, 32, 64, 128, 256, 512, 1024]:
            for kernel in ["L2_1", "L4_2"]:
                for steps in [4, 16, 64, 256]:
                    yield "cartesian", dict(n=n, steps=steps, kernel=kernel)
    elif suite == "performance":
        for n in [32, 64, 128, 256]:
            for p in [1, 2, 3]:
                for fields in [1, 4, 16]:
                    yield "translation", dict(n=n, p=p, steps=16, fields=fields)
        for n in [128, 256, 512, 1024, 2048]:
            for fields in [1, 4, 16]:
                yield "cartesian", dict(n=n, steps=16, kernel="L4_2", fields=fields)
    elif suite == "long_time":
        for n in [16, 32, 64, 128]:
            for p in [1, 2, 3]:
                for steps in [100, 1000, 10000]:
                    yield "translation", dict(n=n, p=p, steps=steps, end=10., profile="gaussian")
    else:
        raise ValueError(suite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True,
                        choices=["translation", "spectrum", "cartesian", "performance", "long_time"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = provenance()
    (args.output / (args.suite+"-provenance.json")).write_text(json.dumps(meta, indent=2)+"\n")
    destination = args.output / (args.suite+".jsonl")
    completed = set()
    if destination.exists():
        for line in destination.read_text().splitlines():
            item = json.loads(line)
            if item.get("status") == "ok":
                completed.add(item["case_id"])
    functions = {"translation": translation_case, "cartesian": cartesian_case, "spectrum": spectrum_case}
    for kind, parameters in cases(args.suite):
        case_id = hashlib.sha256(json.dumps([kind, parameters], sort_keys=True).encode()).hexdigest()[:16]
        if case_id in completed:
            continue
        print(json.dumps({"start": case_id, "kind": kind, **parameters}), flush=True)
        start = time.perf_counter()
        try:
            record = {"status": "ok", "result": functions[kind](**parameters)}
        except Exception:
            record = {"status": "failed", "exception": traceback.format_exc()}
        record.update(case_id=case_id, kind=kind, parameters=parameters,
                      wall_seconds=time.perf_counter()-start, provenance=meta["source_sha256"])
        with destination.open("a") as stream:
            stream.write(json.dumps(record)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        cp.get_default_memory_pool().free_all_blocks()
        print(json.dumps({"done": case_id, "status": record["status"], "seconds": record["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
