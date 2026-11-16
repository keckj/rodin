"""Resolve the remaining auxiliary/time error for four separated FE profiles.

Distributed under the Boost Software License, Version 1.0.
This follow-up is triggered by the error exceeding the refined FE floor.
"""

import argparse
import json
from pathlib import Path

from gpu import cp
from adaptive_study import evolve
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    with args.output.open("w") as stream:
        for naux, steps in ((64, 256), (128, 128), (128, 256)):
            for reaction in (False, True):
                for iterations in (0, 8):
                    parameters = dict(
                        nfe=256,
                        naux=naux,
                        steps=steps,
                        end=0.25,
                        fields=4,
                        spread=0.15,
                        mesh_policy="energy",
                        reaction=reaction,
                        projection_iterations=iterations,
                    )
                    print(json.dumps(parameters), flush=True)
                    result = evolve(**parameters)
                    stream.write(
                        json.dumps(dict(result=result, provenance=meta)) + "\n"
                    )
                    stream.flush()
                    print(
                        json.dumps({k: v for k, v in result.items() if k != "history"}),
                        flush=True,
                    )
                    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
