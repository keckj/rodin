"""Matched FE-mesh comparison under smooth periodic mesh grading.

Distributed under the Boost Software License, Version 1.0.
This study compares global RK4 stepping; it does not represent local-time DG.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from statistics import median

from gpu import cp, Mesh, Space, gaussian
from eulerian_dg import EulerianDG
from fourier_carrier import FourierCarrier, UnitaryCellular
from gpu_graph import EvolutionGraph
from day_study import exact_field
from campaign import provenance


class GradedMesh(Mesh):
    def __init__(self, n, grading=0.0, seed=17):
        super().__init__(n, seed=seed)
        vertex = cp.rint(self.triangles * n).astype(cp.int64) % n
        self.vertex_ids = vertex[:, :, 1] * n + vertex[:, :, 0]
        # Derivative 1+grading*cos(...) stays positive. The shifted minima
        # concentrate triangles around the initial scalar bump.
        offsets = cp.asarray([0.15, 0.05])
        self.triangles = cp.ascontiguousarray(
            self.triangles
            + grading / (2 * cp.pi) * cp.sin(2 * cp.pi * (self.triangles + offsets))
        )
        e = self.triangles[:, 1] - self.triangles[:, 0]
        f = self.triangles[:, 2] - self.triangles[:, 0]
        self.areas = (e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]) / 2
        if float(cp.min(self.areas)) <= 0:
            raise ValueError("Vertex grading inverted a straight triangle")
        assert abs(float(cp.sum(self.areas)) - 1) < 1e-12
        edges = cp.roll(self.triangles, -1, axis=1) - self.triangles
        heights = 2 * self.areas / cp.max(cp.linalg.norm(edges, axis=-1), axis=1)
        self.minimum_height = float(cp.min(heights))
        self.area_ratio = float(cp.max(self.areas) / cp.min(self.areas))


def evolve(
    n=32,
    p=4,
    grading=0.8,
    method="auxiliary",
    steps=32,
    modes=95,
    cfl_scale=4.0,
    seed=17,
    fields=1,
    repeats=1,
    profile=gaussian,
):
    mesh = GradedMesh(n, grading, seed=seed)
    space = Space(mesh, p)
    initial = cp.repeat(space.project(profile), fields, axis=1)
    fields = initial.shape[1]
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    if method == "auxiliary":
        carrier = FourierCarrier(space, modes=modes)
        flow = UnitaryCellular(modes)
        old = carrier.project(initial)
        duration = 1 / (math.pi * steps)
        replay = EvolutionGraph(flow.step, old, duration)
    elif method == "dg":
        flow = EulerianDG(space, flux="upwind")
        steps = math.ceil(flow.norm_bound / (2 * math.pi * cfl_scale))
        duration = 1 / (math.pi * steps)
        replay = EvolutionGraph(flow.step, initial, duration)
    else:
        raise ValueError(method)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        if method == "auxiliary":
            old = carrier.project(initial)
            replay.reset(old)
        else:
            replay.reset(initial)
        for _ in range(steps):
            replay.launch(duration)
        value = replay.result()
        if method == "auxiliary":
            value = initial + carrier.adjoint(value - old)
        cp.cuda.Stream.null.synchronize()
        samples.append(time.perf_counter() - start)
    elapsed = median(samples)
    reference = lambda points, t: exact_field(points, t, True, profile=profile)
    errors = space.errors(value, reference, 0.5)
    mass = float(cp.max(cp.abs(space.mass(value) - space.mass(initial))))
    energy = float(cp.max(space.energy(value) / space.energy(initial)))
    assert mass < 3e-10 and energy <= 1 + 3e-10, (mass, energy)
    probe = None
    if method == "dg":
        random = cp.random.RandomState(825).normal(size=(mesh.cells, 4, space.nb))
        energy0 = space.energy(random)
        maximum = cp.ones(4)

        def monitored(v, dt):
            new = flow.step(v, dt)
            cp.copyto(maximum, cp.maximum(maximum, space.energy(new) / energy0))
            return new

        check = EvolutionGraph(monitored, random, duration)
        maximum.fill(1.0)
        for _ in range(steps):
            check.launch(duration)
        check.result()
        probe = float(cp.max(maximum))
        assert probe < 1 + 1e-8, probe
    best = space.project(reference, 0.5, degree=20)
    floors = space.errors(best, reference, 0.5, degree=20)["l2"]
    return dict(
        n=n,
        p=p,
        grading=grading,
        method=method,
        steps=steps,
        modes=modes,
        cfl_scale=cfl_scale,
        seed=seed,
        minimum_height=mesh.minimum_height,
        area_ratio=mesh.area_ratio,
        static_setup_seconds=setup,
        evolution_seconds=elapsed,
        mass_error=mass,
        energy_ratio=energy,
        random_energy_probe=probe,
        best_projection_error=floors[0],
        best_projection_errors=floors,
        fields=fields,
        evolution_samples=samples,
        repeats=repeats,
        **errors
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "graded.jsonl"
    done = (
        {json.loads(l)["case_id"] for l in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    meta = provenance()
    for n in (16, 32, 64):
        for grading in (0.0, 0.8, 0.95):
            for method in ("auxiliary", "dg"):
                parameters = dict(n=n, grading=grading, method=method)
                key = hashlib.sha256(
                    json.dumps(parameters, sort_keys=True).encode()
                ).hexdigest()[:16]
                if key in done:
                    continue
                print(json.dumps(dict(start=key, **parameters)), flush=True)
                try:
                    record = dict(status="ok", result=evolve(**parameters))
                except Exception:
                    record = dict(status="failed", exception=traceback.format_exc())
                record.update(case_id=key, parameters=parameters, provenance=meta)
                with path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                print(json.dumps(dict(done=key, status=record["status"])), flush=True)
                cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
