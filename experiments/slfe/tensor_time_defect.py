"""Resolve the nonmonotone timestep error by a discrete error decomposition.

Distributed under the Boost Software License, Version 1.0.
An independently evaluated FE projection defines the polynomial reference.
The 2048-step solution is only a checked small-step proxy, not an exact limit.
"""

import argparse
import json
from pathlib import Path

from gpu import cp
from tensor_study import evolve
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    solutions = {}
    records = []
    for steps in (1, 64, 1024, 2048):
        space, value, _, exact, record = evolve(
            n=64, steps=steps, flow="cellular", with_fields=True
        )
        solutions[steps] = value
        records.append(record)
        print(json.dumps(dict(steps=steps, l2=record["l2"])), flush=True)
    projected = space.project(exact, degree=20)
    limit = solutions[2048]
    convergence = float(cp.sqrt(space.energy(solutions[1024] - limit)[0]))
    spatial = limit - projected
    inner = lambda a, b: float(cp.sum(space.mesh.areas[:, None, None] * a * b))
    decomposition = []
    for steps in (1, 64):
        temporal = solutions[steps] - limit
        total = solutions[steps] - projected
        a, b, cross = (
            inner(spatial, spatial),
            inner(temporal, temporal),
            2 * inner(spatial, temporal),
        )
        measured = inner(total, total)
        defect = abs(measured - (a + b + cross))
        assert defect < 1e-16
        decomposition.append(
            dict(
                steps=steps,
                spatial_proxy_squared=a,
                temporal_proxy_squared=b,
                twice_cross_inner=cross,
                total_projected_error_squared=measured,
                identity_defect=defect,
            )
        )
    result = dict(
        small_step_difference_l2=convergence,
        decomposition=decomposition,
        runs=records,
        provenance=provenance(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["decomposition"]), flush=True)


if __name__ == "__main__":
    main()
