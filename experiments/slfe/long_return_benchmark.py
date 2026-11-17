"""Interleaved full GPU return costs for geometry and memory variants.

Distributed under the Boost Software License, Version 1.0.
Every repetition constructs and evolves its own adaptive support. Setup,
controller, rebuilding, transfer, and evolution costs are retained. Error
quadrature is outside the evolution timer, as in the original campaign.
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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    meta = provenance()
    configurations = [
        ("tiled", "galerkin"),
        ("warp", "galerkin"),
        ("tiled", "chebyshev"),
        ("warp", "chebyshev"),
        ("tiled", "block"),
    ]
    if args.pilot:
        configurations = configurations[:1]
    done = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text().splitlines():
            old = json.loads(line)["result"]
            done.add((old["repetition"], old["carrier_builder"], old["method"]))
    completed = 0
    with args.output.open("a" if args.resume else "w") as stream:
        for repetition in range(args.repeats):
            order = configurations[repetition:] + configurations[:repetition]
            if repetition % 2:
                order.reverse()
            for builder, method in order:
                if (repetition, builder, method) in done:
                    continue
                print(
                    json.dumps(
                        dict(repetition=repetition, builder=builder, method=method)
                    ),
                    flush=True,
                )
                result = evolve(
                    nfe=256,
                    naux=128,
                    half_duration=1.0,
                    steps=512,
                    method=method,
                    gram_backend="reduction",
                    carrier_builder=builder,
                )
                result["repetition"] = repetition
                result["complete_seconds"] = (
                    result["setup_seconds"] + result["evolution_seconds"]
                )
                stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
                stream.flush()
                print(
                    json.dumps({k: v for k, v in result.items() if k != "history"}),
                    flush=True,
                )
                cp.get_default_memory_pool().free_all_blocks()
                completed += 1
                if args.limit is not None and completed >= args.limit:
                    return


if __name__ == "__main__":
    main()
