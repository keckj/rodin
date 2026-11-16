"""Empirical RK4 step study with rough-field energy probes and accuracy checks.

Distributed under the Boost Software License, Version 1.0.
Energy probes are finite experiments, not a general stability certificate.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import math

from gpu import cp, Mesh, Space, gaussian
from eulerian_dg import EulerianDG
from gpu_graph import EvolutionGraph
from dg_study import evolve
from campaign import provenance


def probe(n, p, flux, cfl_scale, seed=17):
    space = Space(Mesh(n, seed=seed), p)
    flow = EulerianDG(space, flux=flux)
    random = cp.random.RandomState(924).normal(size=(space.mesh.cells, 4, space.nb))
    initial = cp.concatenate((space.project(gaussian), random), axis=1)
    energy0 = space.energy(initial)
    maximum = cp.ones(5)
    steps = math.ceil(flow.norm_bound / (2 * cp.pi * cfl_scale))
    dt = 1 / (math.pi * steps)

    def monitored_step(value, duration):
        result = flow.step(value, duration)
        cp.copyto(maximum, cp.maximum(maximum, space.energy(result) / energy0))
        return result

    graph = EvolutionGraph(monitored_step, initial, dt)
    maximum.fill(1.0)
    for _ in range(steps):
        graph.launch(dt)
    result = graph.result()
    finite = bool(cp.all(cp.isfinite(result)))
    ratio = float(cp.max(maximum)) if finite else None
    energy_finite = ratio is not None and math.isfinite(ratio)
    stable = finite and energy_finite and ratio <= 1 + 1e-8
    if not energy_finite:
        ratio = None
    return dict(
        n=n,
        p=p,
        flux=flux,
        cfl_scale=cfl_scale,
        seed=seed,
        steps=steps,
        finite=finite,
        energy_finite=energy_finite,
        maximum_energy_ratio=ratio,
        stable_probe=stable,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "dg-cfl.jsonl"
    done = (
        {json.loads(line)["case_id"] for line in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    meta = provenance()
    for flux in ("central", "upwind"):
        for p in (3, 4):
            for n in (16, 32, 64):
                for scale in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0):
                    parameters = dict(n=n, p=p, flux=flux, cfl_scale=scale)
                    key = hashlib.sha256(
                        json.dumps(parameters, sort_keys=True).encode()
                    ).hexdigest()[:16]
                    if key in done:
                        continue
                    result = probe(**parameters)
                    if result["stable_probe"]:
                        result["benchmark"] = evolve(**parameters)
                    record = dict(
                        status="ok", case_id=key, result=result, provenance=meta
                    )
                    with path.open("a") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    print(
                        json.dumps(dict(**parameters, stable=result["stable_probe"])),
                        flush=True,
                    )
                    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
