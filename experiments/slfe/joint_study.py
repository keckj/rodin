"""Combine no-op-consistent memory projection with the minimax periodic phase.

Distributed under the Boost Software License, Version 1.0.
Separate controls already isolate each ingredient; these runs test interaction
and a longer deformation on a finer FE space.
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
        dict(steps=s, check_every=c, floating_phase=True)
        for s, c in ((128, 4), (512, 4), (512, 16))
    ]
    cases += [
        dict(nfe=256, naux=128, half_duration=1.0, steps=512, floating_phase=f)
        for f in (False, True)
    ]
    meta = provenance()
    with args.output.open("w") as stream:
        for parameters in cases:
            print(json.dumps(parameters), flush=True)
            result = evolve(
                method="galerkin", phase_metric="maximum-displacement", **parameters
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
