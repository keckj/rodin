"""Run remaining GPU studies under a deadline, preserving each finished case.

Distributed under the Boost Software License, Version 1.0.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", required=True, help="ISO timestamp including UTC offset")
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/"campaign.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {"pid": os.getpid(), "started": time.time(), "deadline": deadline, "stages": []}
    stages = [
        ("curved_moments", [sys.executable, str(HERE/"research.py"), "--suite", "curved_moments", "--output", str(args.output), "--deadline", str(deadline)]),
        ("support", [sys.executable, str(HERE/"research.py"), "--suite", "support", "--output", str(args.output), "--deadline", str(deadline)]),
        ("robustness", [sys.executable, str(HERE/"research.py"), "--suite", "robustness", "--output", str(args.output), "--deadline", str(deadline)]),
    ]
    for name, command in stages:
        if time.time() >= deadline:
            break
        item = {"name": name, "command": command, "started": time.time()}
        status["current"] = name
        (args.output/"overnight-status.json").write_text(json.dumps(status, indent=2)+"\n")
        with (args.output/(name+".log")).open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=max(1, deadline-time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = "deadline"
        item.update(finished=time.time(), returncode=code)
        status["stages"].append(item)
        # An updated draft is useful even if the next stage stops at the deadline.
        with (args.output/"overnight-report.log").open("a") as log:
            subprocess.run([sys.executable, str(HERE/"report.py"), "--input", str(args.output),
                            "--output", str(args.output/"report")], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, timeout=180)
    status.update(current=None, finished=time.time())
    (args.output/"overnight-status.json").write_text(json.dumps(status, indent=2)+"\n")


if __name__ == "__main__":
    main()
