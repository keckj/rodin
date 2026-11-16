"""Interleave full-setup moving-carrier and DG timing trials on one GPU.

Distributed under the Boost Software License, Version 1.0.
The common FE mesh, initial projection, and reference diagnostics are excluded;
auxiliary centroid detection, all remaps, and the final FE transfer are included.
No failed tolerance case contributes to the reported speed comparison.
"""

import argparse
import json
from pathlib import Path

from gpu import cp
from moving_study import evolve
from night_dg_benchmark import run
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    with args.output.open("w") as stream:
        for fields in (1, 4, 16):
            for repetition in range(3):
                for method in (
                    ("dg", "moving") if repetition % 2 == 0 else ("moving", "dg")
                ):
                    print(
                        json.dumps(
                            dict(
                                start=True,
                                method=method,
                                fields=fields,
                                repetition=repetition,
                            )
                        ),
                        flush=True,
                    )
                    if method == "dg":
                        result = next(run(128, fields, scales=(8.0,), repeats=1))
                    else:
                        result = evolve(nfe=128, naux=64, steps=64, fields=fields)
                    accepted = (
                        result.get("accepted", True) and max(result["l2"]) <= 1e-3
                    )
                    record = dict(
                        method=method,
                        fields=fields,
                        repetition=repetition,
                        tolerance=1e-3,
                        accepted=accepted,
                        result=result,
                        provenance=meta,
                    )
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    print(
                        json.dumps(
                            dict(
                                done=True,
                                method=method,
                                fields=fields,
                                repetition=repetition,
                                accepted=accepted,
                                total_seconds=result["total_seconds"],
                                maximum_l2=max(result["l2"]),
                            )
                        ),
                        flush=True,
                    )
                    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
