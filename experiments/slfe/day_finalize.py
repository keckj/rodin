"""Refresh the daytime report and publish generated artifacts to the allowed fork.

Distributed under the Boost Software License, Version 1.0.
This finalizer stages only generated deliverables, never source modifications.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALLOWED = "https://github.com/keckj/rodin.git"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if git("branch", "--show-current") != "SLFE":
        raise RuntimeError("Expected branch SLFE")
    if args.publish and git("remote", "get-url", "--push", "fork") != ALLOWED:
        raise RuntimeError("Refusing an unapproved remote")
    report = args.input / "day-report"
    report.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(HERE / "day_report.py"),
            "--input",
            str(args.input),
            "--output",
            str(report),
        ],
        cwd=ROOT,
        check=True,
        timeout=240,
    )
    for source in ("memory_theory", "rotation_theory"):
        with (report / (source + "-build.log")).open("w") as log:
            for _ in range(2):
                subprocess.run(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-output-directory=" + str(report),
                        str(HERE / (source + ".tex")),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=120,
                )
    for path in report.iterdir():
        if path.suffix in (".pdf", ".svg", ".html", ".tex", ".json"):
            shutil.copy2(path, HERE / "report_day" / path.name)
    for name in ("ensemble.jsonl", "ensemble-status.json", "ensemble-smoke.jsonl"):
        if (args.input / name).exists():
            shutil.copy2(args.input / name, HERE / "results_day" / name)
    manifest = {
        str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("results_day", "report_day")
        for p in sorted((HERE / folder).rglob("*"))
        if p.is_file()
    }
    (HERE / "day_artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.publish:
        if git("diff", "--cached", "--name-only"):
            raise RuntimeError(
                "Index is not empty; report generated but publication deferred"
            )
        subprocess.run(
            [
                "git",
                "add",
                "experiments/slfe/report_day",
                "experiments/slfe/results_day",
                "experiments/slfe/day_artifacts.json",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(["git", "diff", "--cached", "--check"], cwd=ROOT, check=True)
        if git("diff", "--cached", "--name-only"):
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "Refresh SLFE daytime robustness results and report [skip ci]",
                ],
                cwd=ROOT,
                check=True,
            )
        subprocess.run(
            ["git", "push", ALLOWED, "SLFE:SLFE"], cwd=ROOT, check=True, timeout=180
        )
    receipt = dict(
        finished=datetime.now().astimezone().isoformat(),
        report=str(HERE / "report_day/rapport_journee.pdf"),
        published=args.publish,
        commit=git("rev-parse", "HEAD"),
        remote=ALLOWED if args.publish else None,
    )
    (args.input / "day-delivery.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
