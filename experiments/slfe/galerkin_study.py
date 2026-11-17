"""GPU augmented-Galerkin memory validation and return experiment.

Distributed under the Boost Software License, Version 1.0.
The method is a state-dependent subspace projection, not a linear solver
with a fixed polynomial transfer. Its cost and PDE error remain empirical.
"""

import argparse
import json
from pathlib import Path

from gpu import cp
from warm_validation import validate
from roundtrip_study import evolve
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        validate(p, methods=("galerkin0", "galerkin2", "galerkin4")) for p in (1, 2, 4)
    ]
    meta = provenance()
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(dict(results=records, provenance=meta), indent=2) + "\n"
    )
    with args.output.open("w") as stream:
        for steps, interval in ((128, 4), (512, 4), (512, 16)):
            result = evolve(steps=steps, check_every=interval, method="galerkin")
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
