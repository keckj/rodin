"""Finalize the requested report and publish only the explicitly authorized fork.

Distributed under the Boost Software License, Version 1.0.
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
ALLOWED_REMOTE = "https://github.com/keckj/rodin.git"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    if git("branch", "--show-current") != "SLFE":
        raise RuntimeError("Refusing publication outside branch SLFE")
    report = args.input/"report"
    command = [sys.executable, str(HERE/"report.py"), "--input", str(args.input), "--output", str(report)]
    if not args.draft:
        command.append("--final")
    subprocess.run(command, cwd=ROOT, check=True, timeout=180)
    theory = args.input/"theory"
    theory.mkdir(exist_ok=True)
    with (theory/"build.log").open("w") as log:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory",
                        str(theory), str(HERE/"theory.tex")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=120)
    destination = HERE/"report"
    results = HERE/"results"
    destination.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    for path in report.iterdir():
        if path.suffix in (".pdf", ".svg", ".html", ".json", ".tex"):
            shutil.copy2(path, destination/path.name)
    shutil.copy2(theory/"theory.pdf", destination/"theory.pdf")
    for path in args.input.iterdir():
        if path.suffix in (".json", ".jsonl") and path.name != "delivery.json":
            shutil.copy2(path, results/path.name)
    for name in ["sanitizer.log", "validation.log"]:
        path = args.input/name
        if path.exists():
            shutil.copy2(path, results/(name+".txt"))
    manifest = {"generated_utc": datetime.now().astimezone().isoformat(),
                "source_commit": git("rev-parse", "HEAD"),
                "files": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in [results, destination] for p in folder.iterdir()
                          if p.is_file() and p.name != "manifest.json"}}
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if args.publish:
        # No implicit remote, no upstream PR, no force push, and no source edits
        # staged by this scheduled job. Only generated deliverables are committed.
        if git("remote", "get-url", "--push", "fork") != ALLOWED_REMOTE:
            raise RuntimeError("Fork push URL differs from the user-authorized repository")
        staged = git("diff", "--cached", "--name-only")
        if staged:
            raise RuntimeError("Index already contains staged changes; refusing to include them")
        subprocess.run(["git", "add", "experiments/slfe/report", "experiments/slfe/results"], cwd=ROOT, check=True)
        if git("diff", "--cached", "--name-only"):
            subprocess.run(["git", "commit", "-m", "Record SLFE GPU research report and reproducible results [skip ci]"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", ALLOWED_REMOTE, "SLFE:SLFE"], cwd=ROOT, check=True, timeout=180)
    receipt = {"report": str(report/"rapport.pdf"), "published": args.publish, "final": not args.draft,
               "remote": ALLOWED_REMOTE if args.publish else None,
               "commit": git("rev-parse", "HEAD"), "finished": datetime.now().astimezone().isoformat()}
    (args.input/"delivery.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
