"""Sequential, persistent GPU campaign with explicit failure records.

Distributed under the Boost Software License, Version 1.0.
One child runs at a time. Independent campaigns continue after a failure;
their complete records and exit codes remain available to the report.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    tasks = [
        ("monitor-stationary", "monitor_study.py", ["--suite", "stationary"], "jsonl"),
        ("warm-return", "warm_study.py", [], "jsonl"),
        ("remap-bound-validation", "ambient_validation.py", [], "json"),
        ("compressible", "compressible_study.py", [], "jsonl"),
        ("products", "product_study.py", [], "jsonl"),
        ("svd", "svd_study.py", [], "jsonl"),
        ("source-smoke", "source_study.py", ["--suite", "smoke"], "jsonl"),
        ("source-full", "source_study.py", ["--suite", "full"], "jsonl"),
        ("monitor-return", "monitor_study.py", ["--suite", "return"], "jsonl"),
        (
            "roundtrip-long",
            "roundtrip_study.py",
            ["--suite", "long", "--resume"],
            "jsonl",
        ),
    ]
    records = []
    status = args.output / "queue-status.json"

    def save():
        temporary = status.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(status)

    for name, script, flags, extension in tasks:
        command = [
            sys.executable,
            "-u",
            str(here / script),
            *flags,
            "--output",
            str(args.output / (name + "." + extension)),
        ]
        record = dict(
            name=name,
            command=command,
            status="running",
            started=datetime.now().astimezone().isoformat(),
        )
        records.append(record)
        save()
        print(name, "running", flush=True)
        with (args.output / (name + ".log")).open(
            "a" if "--resume" in flags else "w"
        ) as log:
            completed = subprocess.run(
                command,
                cwd=here.parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record.update(
            status="completed" if completed.returncode == 0 else "failed",
            returncode=completed.returncode,
            finished=datetime.now().astimezone().isoformat(),
        )
        save()
        print(name, record["status"], flush=True)
    subprocess.run(
        [sys.executable, str(here / "night_finalize.py"), "--input", str(args.output)],
        cwd=here.parents[1],
        check=True,
    )


if __name__ == "__main__":
    main()
