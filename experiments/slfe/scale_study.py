"""Resolved and unresolved narrow features on a graded FE mesh.

Distributed under the Boost Software License, Version 1.0.
The test intentionally exposes a fixed auxiliary space's resolution limit.
"""

import argparse
import json
import math
from pathlib import Path
import time
import traceback

from gpu import cp, Space
from graded_study import GradedMesh
from fourier_carrier import FourierCarrier, UnitaryCellular
from eulerian_dg import EulerianDG
from day_study import reference_feet
from gpu_graph import EvolutionGraph
from campaign import provenance


def evolve(
    width=0.003, modes=95, method="auxiliary", n=64, steps=16, end=0.005, factor=1
):
    space = Space(GradedMesh(n, 0.95), 4)

    def profile(points, time):
        r = cp.sin(cp.pi * (points - cp.asarray([0.35, 0.45])))
        return (
            cp.exp(-cp.sum(r * r, axis=-1) / (2 * (cp.pi * width) ** 2))
            / (math.sqrt(math.pi) * width)
        )[..., None]

    def exact(points, time):
        return profile(reference_feet(points, -time, 128), 0)

    initial = space.project(profile, degree=20)
    initial_error = space.errors(initial, profile, 0.0, degree=20, subdivisions=2)[
        "l2"
    ][0]
    best = space.project(exact, end, degree=20)
    best_error = space.errors(best, exact, end, degree=20, subdivisions=2)["l2"][0]
    exact_change = float(
        cp.sqrt(cp.sum(space.mesh.areas[:, None, None] * (best - initial) ** 2))
    )
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    captured = None
    bound = None
    if method == "auxiliary":
        carrier = FourierCarrier(space, modes=modes, factor=factor)
        flow = UnitaryCellular(modes)
        old = carrier.project(initial)
        captured = float(cp.sum(old**2) / modes**2 / space.energy(initial)[0])
        bound = 2 * float(cp.sqrt(cp.sum(old**2) / modes**2))
        replay = EvolutionGraph(flow.step, old, end / steps)
    else:
        flow = EulerianDG(space, flux="upwind")
        steps = math.ceil(end * flow.norm_bound / (2 * 6.0))
        replay = EvolutionGraph(flow.step, initial, end / steps)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(steps):
        replay.launch(end / steps)
    value = replay.result()
    if method == "auxiliary":
        value = initial + carrier.adjoint(value - old)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    error = space.errors(value, exact, end, degree=20, subdivisions=2)
    coarse = space.errors(value, exact, end, degree=20)["l2"][0]
    assert abs(coarse - error["l2"][0]) < max(1e-8, 1e-3 * error["l2"][0]), (
        coarse,
        error["l2"][0],
    )
    energy = float(space.energy(value)[0] / space.energy(initial)[0])
    mass = float(cp.max(cp.abs(space.mass(value) - space.mass(initial))))
    change = float(
        cp.sqrt(cp.sum(space.mesh.areas[:, None, None] * (value - initial) ** 2))
    )
    assert energy < 1 + 3e-10 and mass < 3e-10, (energy, mass)
    if bound is not None:
        assert change < bound + 3e-10, (change, bound)
    return dict(
        width=width,
        factor=factor,
        modes=modes,
        method=method,
        n=n,
        steps=steps,
        end=end,
        initial_error=initial_error,
        best_projection_error=best_error,
        captured_energy_fraction=captured,
        change_upper_bound=bound,
        required_fe_change=exact_change,
        actual_fe_change=change,
        fe_error_lower_bound=(
            max(0.0, exact_change - bound) if bound is not None else None
        ),
        static_setup_seconds=setup,
        evolution_seconds=elapsed,
        energy_ratio=energy,
        mass_error=mass,
        error_without_subdivision=coarse,
        **error
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("limits", "enrichment", "fourier-enrichment"),
        default="limits",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = provenance()
    cases = [
        dict(width=w, method=method, modes=m)
        for w in (0.0015, 0.003, 0.006, 0.012)
        for method, m in [("auxiliary", n) for n in (95, 191, 383, 767)] + [("dg", 95)]
    ]
    if args.suite == "enrichment":
        cases = [
            dict(width=w, factor=f, modes=767)
            for w in (0.0015, 0.003, 0.006)
            for f in (2, 4, 8)
        ]
    elif args.suite == "fourier-enrichment":
        cases = [dict(width=0.0015, factor=f, modes=1535) for f in (4, 8)]
    with (args.output / ("scale-" + args.suite + ".jsonl")).open("w") as f:
        for case in cases:
            try:
                r = dict(status="ok", result=evolve(**case))
            except Exception:
                r = dict(status="failed", error=traceback.format_exc())
            r.update(parameters=case, provenance=meta)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
