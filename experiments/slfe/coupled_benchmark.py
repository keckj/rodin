"""Matched GPU comparison for cellular advection and FE reaction on graded meshes.

Distributed under the Boost Software License, Version 1.0.
Both methods use the same local Galerkin reaction matrices and FE spaces.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

from gpu import cp, Space, gaussian
from graded_study import GradedMesh
from eulerian_dg import EulerianDG
from fourier_carrier import FourierCarrier, UnitaryCellular
from memory_transport import MemoryTransport
from gpu_graph import EvolutionGraph
from reaction_study import Reaction, exact
from campaign import provenance


def evolve(n=32, grading=0.95, method="auxiliary", steps=64, modes=95, cfl_scale=6.0):
    space = Space(GradedMesh(n, grading), 4)
    initial = space.project(gaussian)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    reaction = Reaction(space)
    graph_difference = None
    if method == "auxiliary":
        c = FourierCarrier(space, modes=modes)
        flow = UnitaryCellular(modes)
        dt = 0.5 / steps
        g = reaction.exponential(dt / 2)

        def physics(v):
            return reaction.apply(g, v)

        blas_difference = float(
            cp.max(cp.abs(physics(initial) - cp.einsum("cij,cfj->cfi", g, initial)))
        )
        assert blas_difference < 2e-13, blas_difference
        initial_w = c.project(initial)
        auxiliary_shape = initial_w.shape
        count = initial.size
        packed = cp.concatenate((initial.ravel(), initial_w.ravel()))

        def coupled_step(value, duration):
            # Fixed-step graph: g is the FE exponential for this duration/2.
            v = value[:count].reshape(initial.shape)
            w = value[count:].reshape(auxiliary_shape)
            new = physics(v)
            w = w + c.project(new - v)
            v = new
            new = flow.step(w, duration)
            v = v + c.adjoint(new - w)
            w = new
            new = physics(v)
            w = w + c.project(new - v)
            return cp.concatenate((new.ravel(), w.ravel()))

        replay = EvolutionGraph(coupled_step, packed, dt)
        # Independent execution through the public memory-update operations.
        ordinary = MemoryTransport(c, initial)
        ordinary.apply_physics(physics)
        ordinary.advect(lambda w: flow.step(w, dt))
        ordinary.apply_physics(physics)
        replay.launch(dt)
        check = replay.result()
        graph_difference = max(
            float(
                cp.max(cp.abs(check[:count].reshape(initial.shape) - ordinary.value))
            ),
            float(
                cp.max(
                    cp.abs(check[count:].reshape(auxiliary_shape) - ordinary.auxiliary)
                )
            ),
        )
        assert graph_difference < 2e-11, graph_difference
        replay.reset(packed)
    elif method == "dg":
        flow = EulerianDG(space, flux="upwind")
        flow.diagonal -= reaction.matrix
        # Triangle inequality retains a conservative assembly norm bound.
        bound = flow.norm_bound + float(
            cp.max(cp.sum(cp.abs(reaction.matrix), axis=-1))
        )
        steps = math.ceil(0.5 * bound / (2 * cfl_scale))
        dt = 0.5 / steps
        replay = EvolutionGraph(flow.step, initial, dt)
    else:
        raise ValueError(method)
    cp.cuda.Stream.null.synchronize()
    replay.stream.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(steps):
        replay.launch(dt)
    value = replay.result()
    if method == "auxiliary":
        value = value[: initial.size].reshape(initial.shape)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    errors = space.errors(value, exact, 0.5)
    ratio = float(cp.max(space.energy(value) / space.energy(initial)))
    assert ratio < 1 + 1e-10, ratio
    return dict(
        n=n,
        p=4,
        grading=grading,
        method=method,
        steps=steps,
        modes=modes,
        cfl_scale=cfl_scale,
        static_setup_seconds=setup,
        evolution_seconds=elapsed,
        energy_ratio=ratio,
        graph_difference=graph_difference,
        **errors
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "coupled-benchmark.jsonl"
    done = (
        {json.loads(l)["case_id"] for l in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    meta = provenance()
    for n in (32, 64):
        for grading in (0.0, 0.95):
            for method, steps in (
                ("auxiliary", 32),
                ("auxiliary", 64),
                ("auxiliary", 128),
                ("dg", 64),
            ):
                parameters = dict(n=n, grading=grading, method=method, steps=steps)
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
