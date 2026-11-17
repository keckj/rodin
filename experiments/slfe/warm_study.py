"""GPU return-test comparison of cold and warm memory transfers.

Distributed under the Boost Software License, Version 1.0.
All controls retain the original energy monitor and adaptation threshold.
"""

import argparse
import json
from pathlib import Path

from gpu import cp
from roundtrip_study import evolve
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        dict(steps=steps, check_every=interval, method=method)
        for steps, interval in ((128, 4), (512, 4), (512, 16))
        for method in ("projected", "warm")
    ]
    meta = provenance()
    with args.output.open("w") as stream:
        for case in cases:
            print(json.dumps(case), flush=True)
            result = evolve(**case)
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
