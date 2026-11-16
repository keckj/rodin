"""GPU DG comparison for the transported localized-feature benchmark.

Distributed under the Boost Software License, Version 1.0.
Records every timestep trial, independent errors, random-field growth probes,
and repeated graph timings. The random probes are finite diagnostics, not a
spectral stability certificate. The comparator uses global RK4 stepping.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median
import time

from gpu import cp, Space
from graded_study import GradedMesh
from eulerian_dg import EulerianDG
from gpu_graph import EvolutionGraph
from moving_study import profile_family
from day_study import reference_feet
from campaign import provenance


def run(nfe, fields, scales=(4.0, 6.0, 8.0, 12.0), repeats=3, end=0.25, width=0.01):
    space = Space(GradedMesh(nfe, 0.95), 4)
    profile = profile_family(width, fields)
    exact = lambda points, t: profile(reference_feet(points, -end, 512), 0)
    initial = space.project(profile, degree=20)
    initial_energy = space.energy(initial)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    flow = EulerianDG(space, flux="upwind")
    cp.cuda.Stream.null.synchronize()
    assembly = time.perf_counter() - start
    best = space.errors(
        space.project(exact, degree=20), exact, 0, degree=20, subdivisions=2
    )["l2"]
    probes = cp.random.RandomState(712).normal(size=(space.mesh.cells, 4, space.nb))
    probe_energy = space.energy(probes)
    for scale in scales:
        steps = math.ceil(end * flow.norm_bound / (2 * scale))
        dt = end / steps
        probe = EvolutionGraph(flow.step, probes, dt)
        maximum = 0.0
        for _ in range(64):
            probe.launch(dt)
            ratio = float(cp.max(space.energy(probe.result()) / probe_energy))
            maximum = max(maximum, ratio)
            if not math.isfinite(ratio) or ratio > 1.1:
                break
        del probe
        record = dict(
            nfe=nfe,
            fields=fields,
            width=width,
            end=end,
            cfl_scale=scale,
            steps=steps,
            random_probe_max=maximum,
            assembly_seconds=assembly,
            best_fe_l2=best,
        )
        if maximum > 1.1 or not math.isfinite(maximum):
            yield dict(
                **record,
                accepted=False,
                reason="random probe growth exceeded 10 percent"
            )
            continue
        start = time.perf_counter()
        graph = EvolutionGraph(flow.step, initial, dt)
        cp.cuda.Stream.null.synchronize()
        capture = time.perf_counter() - start
        samples = []
        for _ in range(repeats):
            graph.reset(initial)
            graph.result()
            start = time.perf_counter()
            for _ in range(steps):
                graph.launch(dt)
            value = graph.result()
            samples.append(time.perf_counter() - start)
        errors = space.errors(value, exact, 0, degree=20, subdivisions=2)
        energy = float(cp.max(space.energy(value) / initial_energy))
        mass = float(cp.max(cp.abs(space.mass(value) - space.mass(initial))))
        accepted = (
            all(math.isfinite(x) for x in errors["l2"])
            and energy <= 1.001
            and mass < 1e-8
        )
        yield dict(
            **record,
            accepted=accepted,
            samples_seconds=samples,
            evolution_seconds=median(samples),
            setup_seconds=assembly + capture,
            total_seconds=assembly + capture + median(samples),
            energy_ratio=energy,
            mass_error=mass,
            **errors
        )
        del graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfe", type=int, default=128)
    parser.add_argument("--fields", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument(
        "--scales", type=float, nargs="+", default=[4.0, 6.0, 8.0, 12.0]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    with args.output.open("w") as stream:
        for fields in args.fields:
            print(json.dumps(dict(start=True, nfe=args.nfe, fields=fields)), flush=True)
            for record in run(args.nfe, fields, args.scales, args.repeats):
                stream.write(json.dumps(dict(result=record, provenance=meta)) + "\n")
                stream.flush()
                print(json.dumps(record), flush=True)
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
