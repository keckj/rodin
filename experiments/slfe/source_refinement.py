"""Resolve a newly injected FE source by lossless nested auxiliary refinement.

Distributed under the Boost Software License, Version 1.0.
The trigger measures projection resolution of the FE increment at injection.
It is not an estimator of the subsequent PDE error. No coarsening is applied.
"""

import argparse
import json
from pathlib import Path

from gpu import cp
from source_study import evolve
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        dict(nfe=256, naux=32, width=w, source_tolerance=t)
        for w in (0.003, 0.006)
        for t in (0.002, 0.0002)
    ]
    cases += [dict(nfe=256, naux=128, width=w) for w in (0.003, 0.006)]
    meta = provenance()
    with args.output.open("w") as stream:
        for parameters in cases:
            print(json.dumps(parameters), flush=True)
            result = evolve(**parameters)
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
