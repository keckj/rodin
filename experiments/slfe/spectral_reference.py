"""GPU Fourier/RK4 comparator for prescribed periodic cellular transport.

Distributed under the Boost Software License, Version 1.0.
This baseline exploits a smooth periodic problem; it is not a general FE solver.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time

from gpu import cp, gaussian
from day_study import exact_field
from campaign import provenance
from gpu_graph import EvolutionGraph


class SpectralTransport:
    def __init__(self, n, velocity=None):
        self.n = n
        grid = cp.arange(n, dtype=cp.float64) / n
        x, y = cp.meshgrid(grid, grid, indexing="xy")
        self.points = cp.stack((x, y), axis=-1)
        self.ux = cp.sin(cp.pi * x) ** 2 * cp.sin(2 * cp.pi * y)
        self.uy = -cp.sin(cp.pi * y) ** 2 * cp.sin(2 * cp.pi * x)
        if velocity is not None:
            if velocity.shape != (n, n, 2):
                raise ValueError("Expected a sampled two-dimensional velocity")
            self.ux, self.uy = velocity[..., 0], velocity[..., 1]
        frequency = 2 * cp.pi * cp.fft.fftfreq(n, d=1 / n)
        # The real even-grid Nyquist derivative must vanish for skew symmetry.
        if n % 2 == 0:
            frequency[n // 2] = 0
        self.kx = 1j * frequency[None, :]
        self.ky = 1j * frequency[:, None]

    def rhs(self, value):
        transformed = cp.fft.fft2(cp.stack((value, self.ux * value, self.uy * value)))
        derivatives = cp.fft.ifft2(
            cp.stack(
                (
                    self.kx * transformed[0],
                    self.ky * transformed[0],
                    self.kx * transformed[1] + self.ky * transformed[2],
                )
            )
        ).real
        return -0.5 * (
            self.ux * derivatives[0] + self.uy * derivatives[1] + derivatives[2]
        )

    def step(self, value, dt):
        a = self.rhs(value)
        b = self.rhs(value + (0.5 * dt) * a)
        c = self.rhs(value + (0.5 * dt) * b)
        d = self.rhs(value + dt * c)
        return value + (dt / 6) * (a + 2 * b + 2 * c + d)


def interpolate(value, factor):
    """Fourier interpolation with explicit splitting of even-grid Nyquist modes."""
    n = value.shape[0]
    m = n * factor
    spectrum = cp.fft.fftshift(cp.fft.fft2(value))
    padded = cp.zeros((m, m), dtype=cp.complex128)
    first = (m - n) // 2
    padded[first : first + n, first : first + n] = spectrum
    padded[first, :] *= 0.5
    padded[first + n, :] = padded[first, :]
    padded[:, first] *= 0.5
    padded[:, first + n] = padded[:, first]
    result = cp.fft.ifft2(cp.fft.ifftshift(padded)) * factor**2
    assert float(cp.max(cp.abs(result.imag))) < 2e-12
    return result.real


def validate():
    flow = SpectralTransport(16)
    random = cp.random.RandomState(825)
    a, b = random.normal(size=(2, 16, 16))
    ka, kb = flow.rhs(a), flow.rhs(b)
    skew = float(
        cp.abs(cp.sum(a * kb + ka * b)) / (cp.linalg.norm(a) * cp.linalg.norm(kb))
    )
    constant = float(cp.max(cp.abs(flow.rhs(cp.ones_like(a)))))
    mass = float(cp.abs(cp.mean(ka)))
    interpolation = float(cp.max(cp.abs(interpolate(a, 4)[::4, ::4] - a)))
    assert max(skew, constant, mass, interpolation) < 3e-13
    x, y = flow.points[..., 0], flow.points[..., 1]
    wave = cp.cos(2 * cp.pi * (3 * x + 2 * y))
    fine = cp.arange(64, dtype=cp.float64) / 64
    xx, yy = cp.meshgrid(fine, fine)
    wave_error = float(
        cp.max(cp.abs(interpolate(wave, 4) - cp.cos(2 * cp.pi * (3 * xx + 2 * yy))))
    )
    assert wave_error < 2e-13
    return dict(
        skew_error=skew,
        constant_error=constant,
        mass_derivative=mass,
        interpolation_error=interpolation,
        wave_error=wave_error,
    )


def evolve(n, multiplier=1, end=0.5, reversing=True, graph=False):
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    flow = SpectralTransport(n)
    initial = gaussian(flow.points, 0)
    duration = float(cp.sin(cp.asarray(cp.pi * end)) / cp.pi) if reversing else end
    # ||K|| <= 2*pi*N since ||u_x||, ||u_y|| <= 1. RK4's imaginary-axis
    # interval is [-sqrt(8), sqrt(8)]; use |dt|*||K|| <= 2.
    steps = max(1, math.ceil(abs(duration) * math.pi * n)) * multiplier
    dt = duration / steps
    flow.step(initial, dt)  # Warm allocation, kernels and FFT plans.
    replay = EvolutionGraph(flow.step, initial, dt) if graph else None
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    value = initial
    if graph:
        for _ in range(steps):
            replay.launch(dt)
        value = replay.result()
    else:
        for _ in range(steps):
            value = flow.step(value, dt)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    errors = []
    for factor in (2, 4):
        m = factor * n
        grid = cp.arange(m, dtype=cp.float64) / m
        xx, yy = cp.meshgrid(grid, grid)
        points = cp.stack((xx, yy), axis=-1)
        reference = exact_field(points, end, reversing)
        error = float(cp.sqrt(cp.mean((interpolate(value, factor) - reference) ** 2)))
        errors.append(error)
    mass = float(cp.abs(cp.mean(value) - cp.mean(initial)))
    energy = float(cp.sum(value**2) / cp.sum(initial**2))
    assert mass < 1e-11 and energy <= 1 + 1e-11, (mass, energy)
    return dict(
        method="fourier-rk4",
        n=n,
        steps=steps,
        multiplier=multiplier,
        end=end,
        reversing=reversing,
        graph=graph,
        setup_seconds=setup,
        evolution_seconds=elapsed,
        l2=errors[-1],
        error_grid_refinement=abs(errors[1] - errors[0]),
        mass_error=mass,
        energy_ratio=energy,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / (
        "day-spectral-graph.jsonl" if args.graph else "day-spectral.jsonl"
    )
    done = (
        {tuple(json.loads(line)["case"]) for line in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    meta = provenance()
    checks = validate()
    print(json.dumps(checks), flush=True)
    for reversing in (True, False):
        for n in (16, 32, 64, 128, 256):
            for multiplier in (1, 2, 4):
                case = (n, multiplier, reversing)
                if case in done:
                    continue
                result = evolve(
                    n, multiplier=multiplier, reversing=reversing, graph=args.graph
                )
                record = dict(
                    status="ok",
                    case=case,
                    result=result,
                    validation=checks,
                    provenance=meta,
                )
                with path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                print(json.dumps(result), flush=True)
                cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
