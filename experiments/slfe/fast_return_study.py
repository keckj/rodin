"""Full adaptive return experiments after the weighted-Gram CUDA reduction.

Distributed under the Boost Software License, Version 1.0.
Compare individual and shared augmented memory on the existing short and long
return problems, retaining full setup, controller and geometric rebuild costs.
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
    meta = provenance()
    cases = [
        dict(steps=steps, check_every=check, method=method)
        for steps, check in ((128, 4), (512, 4), (512, 16))
        for method in ("galerkin", "block")
    ]
    cases += [
        dict(nfe=256, naux=128, half_duration=1.0, steps=512, method=method)
        for method in ("galerkin", "block")
    ]
    with args.output.open("w") as stream:
        for options in cases:
            print(json.dumps(options), flush=True)
            result = evolve(**options, gram_backend="reduction")
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
