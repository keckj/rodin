"""Build and publish the night report exclusively to the authorized SLFE fork.

Distributed under the Boost Software License, Version 1.0.
Publication stages only generated artifacts; source changes remain untouched.
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
    report = args.input / "report"
    report.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(HERE / "night_report.py"),
            "--input",
            str(args.input),
            "--output",
            str(report),
        ],
        cwd=ROOT,
        check=True,
        timeout=240,
    )
    with (report / "theory-build.log").open("w") as log:
        for _ in range(2):
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory=" + str(report),
                    str(HERE / "restart_theory.tex"),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
    target = HERE / "report_night2"
    target.mkdir(exist_ok=True)
    for path in report.iterdir():
        if path.suffix in (".pdf", ".svg", ".png", ".html", ".tex", ".json"):
            shutil.copy2(path, target / path.name)
    results = HERE / "results_night2"
    results.mkdir(exist_ok=True)
    for path in args.input.iterdir():
        if path.suffix in (".json", ".jsonl") and path.name != "delivery.json":
            shutil.copy2(path, results / path.name)
        if path.name.endswith("sanitizer.log"):
            (results / (path.name + ".txt")).write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    manifest = {
        str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (target, results)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }
    (HERE / "night2_artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.publish:
        if git("diff", "--cached", "--name-only"):
            raise RuntimeError(
                "Index is not empty; report built but publication deferred"
            )
        subprocess.run(
            [
                "git",
                "add",
                "experiments/slfe/report_night2",
                "experiments/slfe/results_night2",
                "experiments/slfe/night2_artifacts.json",
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
                    "Publish illustrated SLFE adaptive-memory night report [skip ci]",
                ],
                cwd=ROOT,
                check=True,
            )
        subprocess.run(
            ["git", "push", "fork", "SLFE"], cwd=ROOT, check=True, timeout=180
        )
    receipt = dict(
        finished=datetime.now().astimezone().isoformat(),
        published=args.publish,
        commit=git("rev-parse", "HEAD"),
        report=str(target / "rapport_nuit.pdf"),
        remote=ALLOWED if args.publish else None,
    )
    (args.input / "delivery.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
