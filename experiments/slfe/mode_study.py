"""Independent GPU validation and accuracy studies for general Fourier modes.

Distributed under the Boost Software License, Version 1.0.
"""

import argparse
import json
from pathlib import Path
import time
import traceback

from gpu import cp, Space, gaussian
from graded_study import GradedMesh
from fourier_modes import FourierModeFlow, CELLULAR, GENERAL
from fourier_carrier import FourierCarrier, UnitaryCellular
from gpu_graph import EvolutionGraph
from campaign import provenance


def validate():
    n = 15
    value = cp.random.RandomState(1729).normal(size=(n, n, 3))
    general = FourierModeFlow(n, CELLULAR)
    original = UnitaryCellular(n)
    generator_error = float(
        cp.max(cp.abs(general.generator(value) - original.generator(value)))
    )
    step_error = float(
        cp.max(cp.abs(general.step(value, 0.03) - original.step(value, 0.03)))
    )
    assert max(generator_error, step_error) < 2e-12, (generator_error, step_error)
    probes = []
    for modes, mean in (
        (GENERAL, (0.0, 0.0)),
        (GENERAL, (0.07, -0.03)),
        ((), (0.2, 0.1)),
        (((0, -2, 0.1, 0.3),), (0.0, 0.0)),
    ):
        flow = FourierModeFlow(n, modes, mean)
        new = flow.step(value, 0.37)
        inverse = flow.step(new, -0.37)
        mass = float(cp.max(cp.abs(cp.mean(new - value, axis=(0, 1)))))
        energy = float(
            cp.max(
                cp.abs(cp.sum(new**2, axis=(0, 1)) / cp.sum(value**2, axis=(0, 1)) - 1)
            )
        )
        reverse = float(cp.max(cp.abs(inverse - value)))
        one = float(cp.max(cp.abs(flow.step(cp.ones_like(value), 0.37) - 1)))
        assert max(mass, energy, reverse, one) < 2e-12, (mass, energy, reverse, one)
        probes.append(
            dict(
                modes=modes,
                mean=mean,
                mass_error=mass,
                energy_error=energy,
                reverse_error=reverse,
                constant_error=one,
            )
        )
    flow = FourierModeFlow(9, GENERAL, (0.07, -0.03))
    eye = cp.eye(81).reshape(9, 9, 81)
    generator = flow.generator(eye).reshape(81, 81)
    skew = float(cp.max(cp.abs(generator + generator.T)))
    matrix = flow.step(eye, 0.3).reshape(81, 81)
    orthogonal = float(cp.max(cp.abs(matrix.T @ matrix - cp.eye(81))))
    assert max(skew, orthogonal) < 2e-12, (skew, orthogonal)
    from cupyx.scipy.linalg import expm

    exact = expm(0.3 * generator)
    temporal = []
    for order in (2, 4):
        errors = []
        for steps in (8, 16, 32, 64):
            state = eye.copy()
            for _ in range(steps):
                state = flow.step(state, 0.3 / steps, order)
            errors.append(float(cp.linalg.norm(state.reshape(81, 81) - exact)))
        ratio = errors[-2] / errors[-1]
        assert 3.8 < ratio < 4.2 if order == 2 else 14.5 < ratio < 17.5, (
            order,
            errors,
            ratio,
        )
        temporal.append(dict(order=order, errors=errors, ratio=ratio))
    points = cp.random.RandomState(3).uniform(size=(10000, 2))
    reference = float(
        cp.max(cp.abs(flow.feet(points, 0.5, 512) - flow.feet(points, 0.5, 1024)))
    )
    assert reference < 2e-10, reference
    replay = EvolutionGraph(flow.step, value[:9, :9].copy(), 0.03)
    replay.launch(0.03)
    graph = float(
        cp.max(cp.abs(replay.result() - flow.step(value[:9, :9].copy(), 0.03)))
    )
    assert graph < 2e-12, graph
    return dict(
        cellular_generator_error=generator_error,
        cellular_step_error=step_error,
        probes=probes,
        skew_error=skew,
        orthogonal_error=orthogonal,
        temporal=temporal,
        reference_difference=reference,
        graph_error=graph,
    )


def evolve(n=32, p=4, grading=0.0, modes=95, steps=32, end=0.5):
    space = Space(GradedMesh(n, grading), p)
    initial = space.project(gaussian)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    carrier = FourierCarrier(space, modes=modes)
    flow = FourierModeFlow(modes, GENERAL, (0.07, -0.03))
    old = carrier.project(initial)
    replay = EvolutionGraph(flow.step, old, end / steps)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    old = carrier.project(initial)
    replay.reset(old)
    for _ in range(steps):
        replay.launch(end / steps)
    value = initial + carrier.adjoint(replay.result() - old)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start

    def reference(points, t):
        return gaussian(flow.feet(points, t, 512), 0)

    errors = space.errors(value, reference, end)
    best = space.project(reference, end, degree=20)
    floor = space.errors(best, reference, end, degree=20)["l2"][0]
    mass = float(cp.max(cp.abs(space.mass(value) - space.mass(initial))))
    energy = float(cp.max(space.energy(value) / space.energy(initial)))
    assert mass < 3e-10 and energy < 1 + 3e-10, (mass, energy)
    return dict(
        n=n,
        p=p,
        grading=grading,
        modes=modes,
        steps=steps,
        end=end,
        velocity_modes=GENERAL,
        mean_velocity=flow.mean,
        static_setup_seconds=setup,
        evolution_seconds=elapsed,
        mass_error=mass,
        energy_ratio=energy,
        best_projection_error=floor,
        **errors
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=("validation", "accuracy", "resolution"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / ("modes-" + args.suite + ".jsonl")
    jobs = (
        [{}]
        if args.suite == "validation"
        else [
            dict(n=n, grading=g, modes=m, steps=s)
            for n in (16, 32, 64)
            for g in (0.0, 0.95)
            for m in (95, 191)
            for s in (16, 32, 64)
        ]
    )
    if args.suite == "resolution":
        jobs = [
            dict(n=64, grading=g, modes=m, steps=s)
            for g in (0.0, 0.95)
            for m in (383, 511, 767)
            for s in (64, 128)
        ]
    done = (
        {json.loads(l)["case_id"] for l in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    meta = provenance()
    with path.open("a") as f:
        for case in jobs:
            key = json.dumps(case, sort_keys=True)
            if key in done:
                continue
            try:
                record = dict(
                    status="ok",
                    result=validate() if args.suite == "validation" else evolve(**case),
                )
            except Exception:
                record = dict(status="failed", error=traceback.format_exc())
            record.update(case_id=key, parameters=case, provenance=meta)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
