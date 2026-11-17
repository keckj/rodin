"""GPU monitor trajectories from transported positive material energy samples.

Distributed under the Boost Software License, Version 1.0.
No FE or DG evolution and no field remapping enter this diagnostic. For the
incompressible prescribed flow, the material energy weights stay fixed.
"""

import argparse
import json
from pathlib import Path

from gpu import cp
from moving_study import profile_family
from energy_mesh import partitions
from day_study import reference_feet
from campaign import provenance


def evolve(samples=512, shifted=False, floating_phase=False):
    coordinate = (cp.arange(samples) + 0.5) / samples
    yy, xx = cp.meshgrid(coordinate, coordinate, indexing="ij")
    points = cp.stack((xx, yy), axis=-1)
    weights = (
        cp.sum(profile_family(0.01, 4, spread=0.15)(points, 0) ** 2, axis=-1)
        / samples**2
    )
    origins = cp.asarray([0.85, 0.95, 0.3, 0.4]) if shifted else cp.zeros(4)
    edges, oriented = partitions(points, weights, 64, origins=origins)
    history = []
    for step in range(1, 33):
        points = reference_feet(points, 0.25 / 32, 16)
        previous = cp.concatenate((edges, oriented)) if floating_phase else None
        next_edges, next_oriented = partitions(
            points, weights, 64, origins=origins, previous_edges=previous
        )
        displacement = cp.max(
            cp.abs(cp.concatenate((next_edges - edges, next_oriented - oriented))),
            axis=1,
        )
        changed = float(displacement.max()) > 0.01
        history.append(
            dict(
                time=step * 0.25 / 32,
                axis_displacement=displacement.tolist(),
                changed=changed,
                edges=next_edges.tolist(),
                oriented=next_oriented.tolist(),
            )
        )
        if changed:
            edges, oriented = next_edges, next_oriented
    return dict(
        samples=samples,
        shifted=shifted,
        floating_phase=floating_phase,
        origins=origins.tolist(),
        changes=sum(x["changed"] for x in history),
        history=history,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floating-phase", action="store_true")
    args = parser.parse_args()
    meta = provenance()
    with args.output.open("w") as stream:
        for n in (512, 1024):
            for shifted in (False,) if args.floating_phase else (False, True):
                result = evolve(n, shifted, args.floating_phase)
                stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
                stream.flush()
                print(n, shifted, result["changes"], flush=True)
                cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
