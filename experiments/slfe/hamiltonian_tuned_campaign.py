"""Sequential GPU baseline tuning and alternated matched timing blocks.

Distributed under the Boost Software License, Version 1.0.
Select the largest tested step scale passing the random-energy probe for
each flux, then compare both selected DG methods with four SLFE cases.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    records = []

    def run(name, arguments):
        command = [
            sys.executable,
            str(here / "hamiltonian_benchmark.py"),
            "--sizes",
            "128",
            "--output",
            str(args.output / (name + ".jsonl")),
        ] + arguments
        with (args.output / (name + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        records.append(dict(name=name, command=command, returncode=result.returncode))
        temporary = args.output / "hamiltonian-tuned-status.tmp"
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(args.output / "hamiltonian-tuned-status.json")
        if result.returncode:
            raise RuntimeError(f"{name} failed: {result.returncode}")

    run(
        "hamiltonian-dg-central-fine",
        ["--methods", "dg-central", "--scales", "5.25", "5.5", "5.75"],
    )
    run(
        "hamiltonian-dg-upwind-fine",
        ["--methods", "dg-upwind", "--scales", "7.25", "7.5", "7.75"],
    )
    candidates = []
    for name in (
        "hamiltonian-dg-refinement",
        "hamiltonian-dg-central-fine",
        "hamiltonian-dg-upwind-fine",
    ):
        candidates += [
            json.loads(line)["result"]
            for line in (args.output / (name + ".jsonl")).read_text().splitlines()
        ]
    chosen = {
        method: max(
            x["parameter"]
            for x in candidates
            if x["method"] == method and x["stable_probe"]
        )
        for method in ("dg-central", "dg-upwind")
    }
    (args.output / "hamiltonian-tuned-protocol.json").write_text(
        json.dumps(
            dict(
                selection="Largest tested step scale passing the full-duration random-energy probe, separately for each flux; empirical, not a stability theorem.",
                chosen_scales=chosen,
                random_seed=631,
                repetitions=5,
                order="Forward on even blocks, reversed on odd blocks; one timed trajectory per case and block.",
            ),
            indent=2,
        )
        + "\n"
    )
    for repeat in range(5):
        arguments = [
            "--sl-steps",
            "4",
            "8",
            "--central-scales",
            str(chosen["dg-central"]),
            "--upwind-scales",
            str(chosen["dg-upwind"]),
            "--scheme",
            "yoshida6",
            "--repeats",
            "1",
        ]
        if repeat % 2:
            arguments += ["--reverse-case-order"]
        run(f"hamiltonian-tuned-matched-{repeat}", arguments)


if __name__ == "__main__":
    main()
