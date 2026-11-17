"""GPU controls and return tests for floating periodic equidistribution phase.

Distributed under the Boost Software License, Version 1.0.
Only the quantile phase changes relative to the fixed-cut monitor control.
"""

import argparse
import json
from pathlib import Path

from gpu import cp
from energy_mesh import partitions
from roundtrip_study import evolve
from campaign import provenance


def validate():
    coordinate = (cp.arange(512) + 0.5) / 512
    yy, xx = cp.meshgrid(coordinate, coordinate, indexing="ij")
    points = cp.stack((xx, yy), axis=-1)
    phases = cp.asarray([0.13, -0.17, 0.28, 0.44])
    previous = phases[:, None] + cp.linspace(0, 1, 33)[None, :]
    main, oriented = partitions(points, cp.ones(xx.shape), 32, previous_edges=previous)
    uniform_error = float(cp.max(cp.abs(cp.concatenate((main, oriented)) - previous)))
    weights = cp.exp(
        5 * cp.cos(2 * cp.pi * (xx - 0.45)) + 3 * cp.cos(2 * cp.pi * (yy - 0.37))
    )
    a, b = partitions(points, weights, 32)
    original = cp.concatenate((a, b))
    a, b = partitions(points, weights, 32, previous_edges=original)
    stationary_error = float(cp.max(cp.abs(cp.concatenate((a, b)) - original)))
    a, b = partitions(points, weights, 32, previous_edges=previous)
    aligned = cp.concatenate((a, b))
    c, d = partitions(
        points, weights, 32, origins=cp.full(4, 0.25), previous_edges=previous
    )
    cut_error = float(cp.max(cp.abs(cp.concatenate((c, d)) - aligned)))
    assert max(uniform_error, stationary_error, cut_error) < 2e-11
    return dict(
        uniform_error=uniform_error,
        stationary_error=stationary_error,
        cut_shift_error=cut_error,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    validation = validate()
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(dict(result=validation, provenance=meta), indent=2) + "\n"
    )
    with args.output.open("w") as stream:
        for steps, interval in ((128, 4), (512, 4), (512, 16)):
            result = evolve(steps=steps, check_every=interval, floating_phase=True)
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
