"""Independent-profile and mesh-seed robustness campaign, GPU workers only.

Distributed under the Boost Software License, Version 1.0.
Each worker uses a fresh CUDA context. The parent schedules and records cases;
all numerical profiles, projections, evolution and diagnostics run on CUDA.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def compare(n=64, seed=17, fields=4, tolerance=1e-5, repeats=3):
    from gpu import cp
    from graded_study import evolve
    from campaign import provenance

    rng = cp.random.RandomState(1000 + seed)
    centers = rng.uniform(0.15, 0.85, size=(fields, 2))

    def profile(points, time):
        distances = cp.sin(cp.pi * (points[..., None, :] - centers))
        return cp.exp(-12 * cp.sum(distances**2, axis=-1))

    attempts = []
    chosen = {}
    configs = [dict(method="dg", cfl_scale=s) for s in (8.0, 6.0, 4.0)]
    configs += [
        dict(method="auxiliary", modes=m, steps=s)
        for m, s in ((95, 16), (95, 32), (127, 32), (191, 32), (191, 64))
    ]
    for config in configs:
        method = config["method"]
        if method in chosen:
            continue
        try:
            result = evolve(
                n=n, seed=seed, grading=0.95, profile=profile, repeats=repeats, **config
            )
            acceptable = max(result["l2"]) <= tolerance
            attempts.append(dict(status="ok", acceptable=acceptable, result=result))
            if acceptable:
                chosen[method] = result
            # Time refinement cannot improve this mesh's best approximation.
            if max(result["best_projection_errors"]) > tolerance:
                break
        except Exception:
            attempts.append(
                dict(status="failed", parameters=config, error=traceback.format_exc())
            )
    return dict(
        n=n,
        seed=seed,
        fields=fields,
        tolerance=tolerance,
        repeats=repeats,
        centers=centers.get().tolist(),
        attempts=attempts,
        chosen=chosen,
        evolution_ratio=(
            (
                chosen["dg"]["evolution_seconds"]
                / chosen["auxiliary"]["evolution_seconds"]
            )
            if len(chosen) == 2
            else None
        ),
        total_ratio=(
            (
                (
                    chosen["dg"]["static_setup_seconds"]
                    + chosen["dg"]["evolution_seconds"]
                )
                / (
                    chosen["auxiliary"]["static_setup_seconds"]
                    + chosen["auxiliary"]["evolution_seconds"]
                )
            )
            if len(chosen) == 2
            else None
        ),
        provenance=provenance(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", help="ISO timestamp with timezone")
    parser.add_argument("--worker", help="JSON parameters for one isolated GPU worker")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        try:
            record = dict(status="ok", result=compare(**json.loads(args.worker)))
        except Exception:
            record = dict(status="failed", error=traceback.format_exc())
        (args.output / "ensemble-worker.json").write_text(json.dumps(record) + "\n")
        return
    deadline = (
        datetime.fromisoformat(args.deadline).timestamp()
        if args.deadline
        else float("inf")
    )
    jobs = (
        [dict(n=16, seed=17, fields=2, tolerance=0.02, repeats=1)]
        if args.smoke
        else [
            dict(n=64, seed=seed, fields=fields)
            for seed in range(17, 33)
            for fields in (1, 4, 16)
        ]
        + [
            dict(n=128, seed=seed, fields=fields)
            for seed in range(17, 21)
            for fields in (1, 4, 16)
        ]
    )
    path = args.output / ("ensemble-smoke.jsonl" if args.smoke else "ensemble.jsonl")
    done = (
        {json.loads(l)["case_id"] for l in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    status = dict(
        started=datetime.now().astimezone().isoformat(),
        pid=os.getpid(),
        planned=len(jobs),
        completed=len(done),
        deadline=args.deadline,
    )
    for case in jobs:
        key = json.dumps(case, sort_keys=True)
        if key in done:
            continue
        if time.time() >= deadline:
            break
        status["current"] = case
        (args.output / "ensemble-status.json").write_text(
            json.dumps(status, indent=2) + "\n"
        )
        command = [
            sys.executable,
            str(HERE / "ensemble_study.py"),
            "--output",
            str(args.output),
            "--worker",
            json.dumps(case),
        ]
        with (args.output / "ensemble-worker.log").open("w") as log:
            try:
                run = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=min(900, max(1, deadline - time.time())),
                )
                record = (
                    json.loads((args.output / "ensemble-worker.json").read_text())
                    if run.returncode == 0
                    else dict(
                        status="failed",
                        error=f"Worker exit {run.returncode}: "
                        + (args.output / "ensemble-worker.log").read_text()[-8000:],
                    )
                )
            except subprocess.TimeoutExpired:
                record = dict(status="failed", error="Worker time limit reached")
        record.update(case_id=key, parameters=case)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        status["completed"] += 1
        print(
            json.dumps(
                dict(
                    case=case,
                    status=record["status"],
                    ratio=record.get("result", {}).get("evolution_ratio"),
                )
            ),
            flush=True,
        )
    status.update(current=None, finished=datetime.now().astimezone().isoformat())
    (args.output / "ensemble-status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
