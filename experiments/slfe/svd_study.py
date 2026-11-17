"""GPU transport from sampled nonseparable streamfunctions and RK4 comparison.

Distributed under the Boost Software License, Version 1.0.
Both schemes use the same periodic nodal grid and CUDA graph replay.
Reported RMS norms do not replace independent finite-element quadrature.
"""

import argparse
import json
import math
from pathlib import Path
import time

from gpu import cp, HERE, launch
from hamiltonian_commutator import ProductHamiltonianFlow
from spectral_reference import SpectralTransport
from gpu_graph import EvolutionGraph
from product_study import profile
from campaign import provenance


def sampled_velocity(n):
    yy, xx = cp.meshgrid(cp.arange(n) / n, cp.arange(n) / n, indexing="ij")
    psi = 0.03 * cp.exp(
        4 * (cp.cos(2 * cp.pi * (xx - 0.5)) + cp.cos(2 * cp.pi * (yy - 0.5)) - 2)
        + 1.2 * (cp.cos(2 * cp.pi * (xx + yy - 1)) - 1)
    )
    frequency = 2j * cp.pi * cp.fft.fftfreq(n, d=1 / n)
    ux = cp.fft.ifft(cp.fft.fft(psi, axis=0) * frequency[:, None], axis=0).real
    uy = -cp.fft.ifft(cp.fft.fft(psi, axis=1) * frequency[None, :], axis=1).real
    return cp.stack((xx, yy), axis=-1), psi, cp.stack((ux, uy), axis=-1)


def reference(points, end=1.0, steps=1024):
    kernel = cp.RawKernel((HERE / "svd_reference.cu").read_text(), "svd_feet")
    flat = cp.ascontiguousarray(points.reshape(-1, 2))
    feet = cp.empty_like(flat)
    launch(kernel, len(flat), (flat, feet, len(flat), -end, steps))
    return profile(feet).reshape(points.shape[:-1])


def validate():
    points, psi, velocity = sampled_velocity(31)
    flow = SpectralTransport(31, velocity)
    random = cp.random.RandomState(693).normal(size=(2, 31, 31))
    a, b = random
    ka, kb = flow.rhs(a), flow.rhs(b)
    skew = float(
        cp.abs(cp.sum(a * kb + b * ka)) / (cp.linalg.norm(a) * cp.linalg.norm(kb))
    )
    constant = float(cp.max(cp.abs(flow.rhs(cp.ones_like(a)))))
    assert max(skew, constant) < 1e-11, (skew, constant)
    points = cp.random.RandomState(167).uniform(0, 1, (16384, 2))
    reference_error = float(
        cp.max(cp.abs(reference(points, steps=512) - reference(points, steps=1024)))
    )
    assert reference_error < 3e-8, reference_error
    return dict(
        skew_error=skew,
        constant_error=constant,
        reference_512_1024_error=reference_error,
    )


def evolve(n=255, method="pcp16", tolerance=1e-6, steps=256, rk_scale=1.0):
    points, psi, velocity = sampled_velocity(n)
    initial = profile(points)[..., None]
    cp.cuda.Stream.null.synchronize()
    before = time.perf_counter()
    approximation = None
    # For the skew form K=-1/2 sum(U_j D_j + D_j U_j), this bounds ||K||_2.
    # RK4 is contractive on the imaginary axis when dt*||K|| <= sqrt(8).
    bound = float(2 * cp.pi * (n // 2) * cp.sum(cp.max(cp.abs(velocity), axis=(0, 1))))
    if method == "pcp16":
        flow = ProductHamiltonianFlow.from_streamfunction(psi, tolerance)
        approximation = flow.approximation
        operation = flow.step
        stages = len(flow.word) * len(flow.stages)
    elif method == "rk4":
        flow = SpectralTransport(n, velocity)
        steps = max(1, math.ceil(bound * rk_scale / 2))
        operation = lambda value, dt: flow.step(value[..., 0], dt)[..., None]
        stages = None
    else:
        raise ValueError("Expected pcp16 or rk4")
    graph = EvolutionGraph(operation, initial, 1 / steps)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - before
    before = time.perf_counter()
    graph.reset(initial)
    for _ in range(steps):
        graph.launch(1 / steps)
    value = graph.result()
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - before
    exact = reference(points)[..., None]
    mass_error = float(cp.abs(cp.mean(value - initial)))
    energy_ratio = float(cp.sum(value**2) / cp.sum(initial**2))
    assert mass_error < 1e-10 and energy_ratio < 1 + 1e-9, (mass_error, energy_ratio)
    return dict(
        n=n,
        method=method,
        tolerance=tolerance,
        steps=steps,
        rk_scale=rk_scale,
        approximation=approximation,
        spectral_norm_bound=bound,
        shears_per_step=stages,
        setup_seconds=setup,
        evolution_seconds=elapsed,
        l2=[float(cp.sqrt(cp.mean((value - exact) ** 2)))],
        metric="nodal RMS",
        mass_error=mass_error,
        energy_ratio=energy_ratio,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = validate()
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    meta = provenance()
    cases = [
        dict(n=n, tolerance=tol, steps=s)
        for n in (127, 255)
        for tol in (1e-2, 1e-4, 1e-6)
        for s in (64, 256)
    ]
    cases += [
        dict(n=n, method="rk4", rk_scale=scale) for n in (127, 255) for scale in (1, 2)
    ]
    with args.output.open("w") as stream:
        for case in cases:
            print(json.dumps(case), flush=True)
            result = evolve(**case)
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(json.dumps(result), flush=True)
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
