"""GPU PDE tests of the graph chart, distinct from the physical L2 memory.

Distributed under the Boost Software License, Version 1.0.
All graph inverses have a fixed polynomial degree and no data-dependent
stopping criterion. The cost and PDE error must be measured anew.
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
    parser.add_argument("--long", action="store_true")
    args = parser.parse_args()
    meta = provenance()
    cases = [
        dict(nfe=128, naux=64, steps=steps, check_every=check, graph_degree=degree)
        for steps, check in ((128, 4), (512, 4), (512, 16))
        for degree in (16, 20)
    ]
    if args.long:
        cases = [
            dict(nfe=256, naux=128, half_duration=1.0, steps=512, graph_degree=degree)
            for degree in (16, 20)
        ]
    with args.output.open("w") as stream:
        for options in cases:
            print(json.dumps(options), flush=True)
            result = evolve(**options, method="graph", carrier_builder="tiled")
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
