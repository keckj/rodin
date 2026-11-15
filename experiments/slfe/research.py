"""Targeted experiments on flow geometry and compatible cubature.

Distributed under the Boost Software License, Version 1.0.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from gpu import cp, Mesh, Space, Transfer, translation, smooth, constant, gaussian, timings
from moments import MomentTransfer
from flows import shear, shear_solution, compressible, compressible_jacobian, compressible_solution, reversible
from campaign import provenance


def deformation(n, p, steps, flow_name, end=1., mode="geometric", subdivisions=3,
                moment_solver="dense", seed=17):
    mesh = Mesh(n, seed=seed)
    space = Space(mesh, p)
    flow = {"shear": shear, "compressible": compressible, "reversible": reversible}[flow_name]
    exact = {"shear": shear_solution, "compressible": compressible_solution,
             "reversible": lambda x, t: gaussian(x, 0)}[flow_name]
    function = gaussian if flow_name == "reversible" else smooth
    c = space.project(function)
    mass0 = space.mass(c)
    dt = end/steps
    setup = 0.
    apply = 0.
    diagnostics = []
    transfer = None
    for k in range(steps):
        # Autonomous shear/compression have the same step map every time.
        if transfer is None or flow_name == "reversible":
            if mode == "geometric":
                moved = mesh.mapped(flow, k*dt, (k+1)*dt)
                transfer = Transfer(space, moved, slots=256)
            else:
                transfer = MomentTransfer(space, flow, k*dt, (k+1)*dt,
                                          fit="positive", solver=moment_solver, qdegree=12,
                                          subdivisions=subdivisions,
                                          jacobian=compressible_jacobian if flow_name == "compressible" else None)
                diagnostics.append(transfer.summary())
            setup += transfer.build_seconds
        start, finish = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        c = transfer.apply(c)
        finish.record()
        finish.synchronize()
        apply += cp.cuda.get_elapsed_time(start, finish)/1000
    return {"n": n, "p": p, "steps": steps, "flow": flow_name, "mode": mode,
            "end": end, "seed": seed, "subdivisions": subdivisions,
            "setup_seconds": setup, "apply_seconds": apply,
            "mass_absolute_error": cp.abs(space.mass(c)-mass0).get().tolist(),
            "accepted_moments": all(x["relative_moment_residual"] < 1e-9 for x in diagnostics),
            "accepted_positive": all(x["minimum_weight_ratio"] >= -1e-12 for x in diagnostics),
            "moment_diagnostics": diagnostics, **space.errors(c, exact, end)}


def moment_case(n, p, subdivisions, fit="positive", solver="dense", courant=.731,
                seed=17, steps=16, qdegree=12, moment_degree=None):
    mesh = Mesh(n, seed=seed)
    space = Space(mesh, p)
    dt = courant/n
    transfer = MomentTransfer(space, translation, 0, dt, subdivisions=subdivisions,
                              fit=fit, solver=solver, qdegree=qdegree, maxiter=4000,
                              active_passes=12, moment_degree=moment_degree)
    one = cp.zeros((mesh.cells, 1, space.nb), dtype=cp.float64)
    one[:, :, 0] = 1.
    random = cp.random.RandomState(seed+1).normal(size=one.shape)
    random_mass_error = float(cp.max(cp.abs(space.mass(transfer.apply(random))-space.mass(random))))
    constant_error = float(cp.max(cp.abs(transfer.apply(one)-one)))
    energy_ratio = float(cp.max(space.energy(transfer.apply(random))/space.energy(random)))
    c = space.project(smooth)
    geometric = Transfer(space, mesh.mapped(translation, 0, dt))
    reference = c.copy()
    for _ in range(steps):
        c = transfer.apply(c)
        reference = geometric.apply(reference)
    record = {"n": n, "p": p, "subdivisions": subdivisions, "courant": courant,
              "seed": seed, "steps": steps, "dt": dt, "moment_degree": moment_degree,
              **transfer.summary(), "constant_error": constant_error,
              "random_mass_error": random_mass_error, "random_energy_ratio": energy_ratio,
              "accepted_moments": transfer.relative_moment_residual < 1e-9,
              "accepted_positive": transfer.minimum_weight_ratio >= -1e-12,
              "l2_geometric": space.errors(reference, smooth, steps*dt)["l2"][0],
              **space.errors(c, smooth, steps*dt)}
    crossed = transfer.source != transfer.target
    incident = cp.zeros(mesh.cells, dtype=cp.int32)
    cp.add.at(incident, transfer.source[crossed], 1)
    cp.add.at(incident, transfer.target[crossed], 1)
    record["crossing_fraction"] = float(cp.mean(crossed))
    record["isolated_cells"] = int(cp.sum(incident == 0))
    # Independent full weighted singular-value check on genuinely small systems.
    if mesh.cells*space.nb <= 800:
        ndof = mesh.cells*space.nb
        matrix = cp.zeros((ndof, ndof), dtype=cp.float64)
        sb, tb = transfer.sb, transfer.tb
        row = transfer.target[:, None, None]*space.nb+cp.arange(space.nb)[None, :, None]
        col = transfer.source[:, None, None]*space.nb+cp.arange(space.nb)[None, None, :]
        block = (transfer.weights/mesh.areas[transfer.target])[:, None, None]*tb[:, :, None]*sb[:, None, :]
        cp.add.at(matrix, (cp.broadcast_to(row, block.shape).ravel(), cp.broadcast_to(col, block.shape).ravel()), block.ravel())
        root = cp.repeat(cp.sqrt(mesh.areas), space.nb)
        singular = cp.linalg.svd(root[:, None]*matrix/root[None, :], compute_uv=False)
        record["singular_max"] = float(singular[0])
        record["singular_above_one"] = int(cp.sum(singular > 1+1e-9))
    return record


def cases(suite):
    if suite == "deformation":
        for flow in ["shear", "compressible", "reversible"]:
            for n in [8, 16, 32, 64, 128]:
                for p in [1, 2, 3]:
                    for steps in [4, 16, 64]:
                        yield "deformation", dict(n=n, p=p, steps=steps, flow_name=flow)
    elif suite == "moments":
        for p, n in [(1, 4), (1, 8), (1, 12), (2, 4), (2, 6), (2, 8)]:
            for subdivisions in [1, 2, 3, 4]:
                for fit in ["raw", "signed", "positive"]:
                    yield "moment", dict(n=n, p=p, subdivisions=subdivisions, fit=fit)
        for p in [1, 2]:
            for n in [4, 8, 16, 32]:
                yield "moment", dict(n=n, p=p, subdivisions=3, solver="cg")
    elif suite == "curved_moments":
        for flow in ["shear", "compressible"]:
            for n in [4, 6, 8]:
                for p in [1, 2]:
                    for steps in [1, 4, 16]:
                        yield "deformation", dict(n=n, p=p, steps=steps, flow_name=flow,
                                                  mode="moments", subdivisions=4)
    elif suite == "robustness":
        for seed in range(1, 41):
            for p, n in [(1, 8), (2, 6)]:
                for courant in [.031, .173, .731, 1.37, 4.13]:
                    for subdivisions in [2, 3, 4]:
                        yield "moment", dict(n=n, p=p, subdivisions=subdivisions,
                                             courant=courant, seed=seed)
    elif suite == "support":
        for p in [1, 2]:
            for courant in [.1, .01, .001, .0001]:
                for subdivisions in [1, 2, 3, 4, 6]:
                    yield "moment", dict(n=4, p=p, subdivisions=subdivisions,
                                         courant=courant, steps=1)
        for p in [1, 2]:
            for moment_degree in [p, 2*p]:
                for courant in [.173, .731, 1.37]:
                    yield "moment", dict(n=8, p=p, subdivisions=3,
                                         courant=courant, moment_degree=moment_degree)
    else:
        raise ValueError(suite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=["deformation", "moments", "curved_moments", "robustness", "support"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, default=float("inf"), help="Unix time; stop between cases")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = provenance()
    (args.output/(args.suite+"-provenance.json")).write_text(json.dumps(meta, indent=2)+"\n")
    destination = args.output/(args.suite+".jsonl")
    completed = set()
    if destination.exists():
        for line in destination.read_text().splitlines():
            try:
                completed.add(json.loads(line)["case_id"])
            except json.JSONDecodeError:
                # Preserve an interrupted record, but separate it from the next.
                continue
        with destination.open("ab") as stream:
            stream.write(b"\n")
    functions = {"deformation": deformation, "moment": moment_case}
    for kind, parameters in cases(args.suite):
        case_id = hashlib.sha256(json.dumps([kind, parameters], sort_keys=True).encode()).hexdigest()[:16]
        if case_id in completed:
            continue
        if time.time() >= args.deadline:
            print("Deadline reached between cases", flush=True)
            break
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
