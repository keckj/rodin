"""GPU validation and PDE tests of product-streamfunction commutator transport.

Distributed under the Boost Software License, Version 1.0.
This exploratory path broadens prescribed velocities beyond a few Fourier
modes; it does not yet accept arbitrary FE velocity fields or nonlinear flow.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, HERE, launch, Space, Mesh
from hamiltonian_commutator import ProductHamiltonianFlow
from fourier_carrier import FourierCarrier
from gpu_graph import EvolutionGraph
from campaign import provenance


TERMS = ((0.03, 4.0, 0.5, 0.5), (-0.015, 7.0, 0.7, 0.7))


def factors(n, rank):
    coordinate = cp.arange(n) / n
    return cp.stack(
        [
            cp.stack(
                (
                    cp.sqrt(abs(amplitude))
                    * cp.exp(kappa * (cp.cos(2 * cp.pi * (coordinate - cy)) - 1)),
                    cp.sign(amplitude)
                    * cp.sqrt(abs(amplitude))
                    * cp.exp(kappa * (cp.cos(2 * cp.pi * (coordinate - cx)) - 1)),
                )
            )
            for amplitude, kappa, cx, cy in TERMS[:rank]
        ]
    )


def profile(points, t=0):
    delta = cp.sin(cp.pi * (points - cp.asarray([0.64, 0.51])))
    width = 0.04
    return cp.exp(-cp.sum(delta**2, axis=-1) / (2 * (cp.pi * width) ** 2)) / (
        cp.sqrt(cp.pi) * width
    )


def reference(points, end, rank, steps=1024):
    kernel = cp.RawKernel((HERE / "product_reference.cu").read_text(), "product_feet")
    flat = cp.ascontiguousarray(points.reshape(-1, 2))
    result = cp.empty_like(flat)
    terms = cp.asarray(TERMS[:rank], dtype=cp.float64)
    launch(kernel, len(flat), (flat, result, len(flat), terms, rank, -end, steps))
    return profile(result).reshape(points.shape[:-1])


def validate():
    records = []
    for rank in (1, 2):
        n = 9
        flow = ProductHamiltonianFlow(factors(n, rank))
        eye = cp.eye(n * n).reshape(n, n, n * n)
        generator = flow.generator(eye).reshape(n * n, n * n)
        skew_error = float(cp.max(cp.abs(generator + generator.T)))
        eigen, vectors = cp.linalg.eigh(1j * generator)
        errors = []
        for duration in (0.02, 0.01, 0.005, 0.0025):
            exact = ((vectors * cp.exp(-1j * duration * eigen)) @ vectors.conj().T).real
            numerical = flow.step(eye, duration).reshape(n * n, n * n)
            error = float(cp.linalg.norm(numerical - exact))
            unitary_error = float(
                cp.max(cp.abs(numerical.T @ numerical - cp.eye(n * n)))
            )
            constant_error = float(cp.max(cp.abs(cp.sum(numerical, axis=1) - 1)))
            inverse_error = float(
                cp.max(
                    cp.abs(
                        flow.inverse_step(numerical.reshape(n, n, n * n), duration)
                        - eye
                    )
                )
            )
            assert max(skew_error, unitary_error, constant_error, inverse_error) < 3e-11
            errors.append(error)
            records.append(
                dict(
                    rank=rank,
                    duration=duration,
                    dense_error=error,
                    skew_error=skew_error,
                    unitary_error=unitary_error,
                    constant_error=constant_error,
                    inverse_error=inverse_error,
                )
            )
        assert 6 < errors[-2] / errors[-1] < 10, errors
    points = cp.random.RandomState(917).uniform(0, 1, (8192, 2))
    reference_error = float(
        cp.max(cp.abs(reference(points, 1.0, 2, 512) - reference(points, 1.0, 2, 1024)))
    )
    assert reference_error < 3e-8, reference_error
    n = 63
    yy, xx = cp.meshgrid(cp.arange(n) / n, cp.arange(n) / n, indexing="ij")
    streamfunction = 0.03 * cp.exp(
        4 * (cp.cos(2 * cp.pi * xx) + cp.cos(2 * cp.pi * yy) - 2)
        + 1.2 * (cp.cos(2 * cp.pi * (xx + yy)) - 1)
    )
    compressed = ProductHamiltonianFlow.from_streamfunction(streamfunction, 1e-6)
    reconstructed = cp.einsum(
        "ry,rx->yx", compressed.factors[:, 0], compressed.factors[:, 1]
    )
    frequencies = 2j * cp.pi * cp.fft.fftfreq(n, d=1 / n)

    def derivative(value, axis):
        multiplier = frequencies[:, None] if axis == 0 else frequencies[None, :]
        return cp.fft.ifft(cp.fft.fft(value, axis=axis) * multiplier, axis=axis).real

    direct_error = float(
        cp.sqrt(
            cp.mean(
                derivative(streamfunction - reconstructed, 0) ** 2
                + derivative(streamfunction - reconstructed, 1) ** 2
            )
        )
    )
    norm = compressed.approximation["velocity_l2"]
    assert direct_error <= 1e-6 * norm * (1 + 1e-7)
    assert (
        abs(direct_error - compressed.approximation["velocity_error_l2"]) < 1e-10 * norm
    )
    return dict(
        records=records,
        reference_512_1024_error=reference_error,
        svd_velocity=dict(**compressed.approximation, direct_error_l2=direct_error),
    )


def evolve(n=127, rank=1, steps=64, method="pcp16", nfe=None, end=1.0):
    flow = ProductHamiltonianFlow(factors(n, rank), method)
    space, carrier = None, None
    yy, xx = cp.meshgrid(cp.arange(n) / n, cp.arange(n) / n, indexing="ij")
    points = cp.stack((xx, yy), axis=-1)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    if nfe is None:
        initial = profile(points)[..., None]
        old = initial
    else:
        space = Space(Mesh(nfe, jitter=0.18, seed=817), 4)
        initial = space.project(profile, degree=20)
        carrier = FourierCarrier(space, modes=n)
        old = carrier.project(initial)
    graph = EvolutionGraph(flow.step, old, end / steps)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    graph.reset(old)
    for _ in range(steps):
        graph.launch(end / steps)
    transported = graph.result()
    value = (
        transported if carrier is None else initial + carrier.adjoint(transported - old)
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    energy_error = float(cp.max(cp.abs(cp.mean(transported**2 - old**2, axis=(0, 1)))))
    mass_error = float(cp.max(cp.abs(cp.mean(transported - old, axis=(0, 1)))))
    assert energy_error < 5e-10 and mass_error < 5e-11, (energy_error, mass_error)
    if space is None:
        exact = reference(points, end, rank)[..., None]
        errors = dict(
            l2=[float(cp.sqrt(cp.mean((value - exact) ** 2)))], metric="nodal RMS"
        )
        floor = None
    else:
        exact = lambda p, t: reference(p, end, rank)
        errors = space.errors(value, exact, 0, degree=20, subdivisions=2)
        errors["metric"] = "independent FE L2 quadrature"
        floor = space.errors(
            space.project(exact, degree=20), exact, 0, degree=20, subdivisions=2
        )["l2"]
    return dict(
        n=n,
        nfe=nfe,
        rank=rank,
        steps=steps,
        method=method,
        end=end,
        shears_per_step=len(flow.word) * len(flow.stages),
        setup_seconds=setup,
        evolution_seconds=elapsed,
        mass_error=mass_error,
        auxiliary_energy_error=energy_error,
        best_fe_l2=floor,
        **errors
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = validate()
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(dict(result=validation, provenance=provenance()), indent=2) + "\n"
    )
    meta = provenance()
    cases = [
        dict(n=127, rank=rank, steps=s, method=m)
        for rank in (1, 2)
        for m in ("group4", "pcp16")
        for s in (16, 32, 64, 128, 256)
    ]
    cases += [
        dict(n=255, rank=rank, steps=s, nfe=nfe)
        for rank in (1, 2)
        for nfe in (32, 64)
        for s in (32, 128, 512)
    ]
    with args.output.open("w") as stream:
        for parameters in cases:
            print(json.dumps(parameters), flush=True)
            result = evolve(**parameters)
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(json.dumps(result), flush=True)
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
