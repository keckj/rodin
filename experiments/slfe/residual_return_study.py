"""Compare physical-increment and fitted-residual seeds in memory compression.

Distributed under the Boost Software License, Version 1.0.
The projection itself and its energy identity are unchanged; only the small
state-dependent search space changes. All variants use the tiled GPU geometry.
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
        dict(steps=steps, check_every=check)
        for steps, check in ((128, 4), (512, 4), (512, 16))
    ]
    cases.append(dict(nfe=256, naux=128, half_duration=1.0, steps=512))
    with args.output.open("w") as stream:
        for options in cases:
            for seed in ("moment", "increment", "residual"):
                print(json.dumps(dict(seed=seed, **options)), flush=True)
                result = evolve(
                    **options,
                    method="galerkin",
                    gram_backend="reduction",
                    carrier_builder="tiled",
                    krylov_seed=seed
                )
                stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
                stream.flush()
                print(
                    json.dumps({k: v for k, v in result.items() if k != "history"}),
                    flush=True,
                )
                cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
