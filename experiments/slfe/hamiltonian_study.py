"""GPU transport driven by Fourier moments of an unstructured FE streamfunction.

Distributed under the Boost Software License, Version 1.0.
Nine integer orientations exercise general area-preserving coordinate
charts. Independent RK4 characteristics use the original analytic velocity;
the error of the filtered FE velocity is measured separately. The support
is uniform and the retained Fourier set is fixed in this experiment.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, HERE, Mesh, Space, launch
from carrier import Carrier
from tensor_hamiltonian import (
    TensorHamiltonian,
    ComposedHamiltonian,
    streamfunction_modes,
)
from memory_transport import MemoryTransport
from graph_memory import GraphMemory
from residual_energy import ResidualEnergy
from krylov_study import ErrorOracle
from campaign import provenance


WAVES = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (1, -2), (3, 2)]


def profile(points, time):
    centers = cp.asarray([[0.27, 0.38], [0.67, 0.62]])
    widths = cp.asarray([0.08, 0.11])
    distance = cp.sin(cp.pi * (points[..., None, :] - centers))
    return cp.exp(-cp.sum(distance**2, axis=-1) / (2 * (cp.pi * widths) ** 2))


def velocity(points, coefficients):
    waves = cp.asarray(WAVES)
    phase = 2 * cp.pi * cp.einsum("...i,mi->...m", points, waves)
    derivative = (
        2
        * cp.pi
        * (-cp.sin(phase) * coefficients[:, 0] + cp.cos(phase) * coefficients[:, 1])
    )
    return cp.stack((derivative @ waves[:, 1], -derivative @ waves[:, 0]), axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--steps", nargs="+", type=int, default=[4, 16, 64])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("reset", "memory", "graph"),
        default=["reset", "memory", "graph"],
    )
    parser.add_argument("--velocity-source", choices=("fe", "analytic"), default="fe")
    parser.add_argument("--fields", action="store_true")
    parser.add_argument(
        "--scheme",
        choices=("strang", "triple4", "suzuki4", "yoshida6"),
        default="strang",
    )
    args = parser.parse_args()
    meta = provenance()
    waves = cp.asarray(WAVES, dtype=cp.float64)
    denominator = cp.sum(waves**2, axis=1)
    analytic = cp.stack(
        (0.018 / denominator, 0.009 * (-1.0) ** cp.arange(len(WAVES)) / denominator),
        axis=1,
    )
    modes = cp.ascontiguousarray(cp.concatenate((waves, analytic), axis=1))
    kernel = cp.RawKernel(
        (HERE / "hamiltonian_reference.cu").read_text(), "hamiltonian_reference"
    )
    kernel.compile()
    duration = 0.5

    def feet(points, substeps=512):
        flat = cp.ascontiguousarray(points.reshape(-1, 2))
        result = cp.empty_like(flat)
        launch(
            kernel,
            len(flat),
            (flat, result, len(flat), modes, len(WAVES), -duration, substeps),
        )
        return result.reshape(points.shape)

    def exact(points):
        return profile(feet(points), 0)

    def streamfunction(points, time):
        phase = 2 * cp.pi * cp.einsum("...i,mi->...m", points, waves)
        return cp.sum(
            cp.cos(phase) * analytic[:, 0] + cp.sin(phase) * analytic[:, 1], axis=-1
        )[..., None]

    probe = cp.random.RandomState(163).uniform(size=(4096, 2))
    ref_error = float(cp.max(cp.abs(feet(probe) - feet(probe, 1024))))
    assert ref_error < 1e-9, ref_error
    exact_velocity = velocity(probe, analytic)
    maximum_speed = float(cp.max(cp.abs(exact_velocity)))
    with args.output.open("w") as output:
        for n in args.sizes:
            space = Space(Mesh(n, jitter=0.18, seed=421), 4)
            initial = space.project(profile, degree=20)
            coefficients = (
                streamfunction_modes(
                    space, space.project(streamfunction, degree=20), WAVES
                )
                if args.velocity_source == "fe"
                else analytic
            )
            coefficient_error = float(
                cp.linalg.norm(coefficients - analytic) / cp.linalg.norm(analytic)
            )
            velocity_error = float(
                cp.linalg.norm(velocity(probe, coefficients) - exact_velocity)
                / cp.linalg.norm(exact_velocity)
            )
            oracle = ErrorOracle(space, profile, duration, reference=exact)
            norms = cp.sqrt(
                cp.sum(oracle.reference**2 * oracle.weights[..., None], axis=(0, 1))
            )
            floors = (cp.asarray(oracle.errors(oracle.best)["l2"]) / norms).tolist()
            carrier = Carrier(space, n=2 * n, builder="tiled")
            energy0, mass0 = space.energy(initial), space.mass(initial)
            for steps in args.steps:
                cp.cuda.Stream.null.synchronize()
                begin = time.perf_counter()
                flow = (
                    TensorHamiltonian(carrier, WAVES, coefficients, duration / steps)
                    if args.scheme == "strang"
                    else ComposedHamiltonian(
                        carrier, WAVES, coefficients, duration / steps, args.scheme
                    )
                )
                flow.apply_carrier(carrier.project(initial))
                cp.cuda.Stream.null.synchronize()
                setup = time.perf_counter() - begin
                fields = {}
                for method in args.methods:
                    print(json.dumps(dict(n=n, steps=steps, method=method)), flush=True)
                    state = (
                        GraphMemory(carrier, initial)
                        if method == "graph"
                        else MemoryTransport(carrier, initial)
                    )
                    cp.cuda.Stream.null.synchronize()
                    begin = time.perf_counter()
                    for _ in range(steps):
                        state.advect(flow.apply_carrier)
                        if method == "reset":
                            state.restart(carrier)
                    cp.cuda.Stream.null.synchronize()
                    elapsed = time.perf_counter() - begin
                    ratio = float(cp.max(state.energy() / energy0))
                    mass_error = float(cp.max(cp.abs(space.mass(state.value) - mass0)))
                    assert ratio <= 1 + 1e-9 and mass_error < 1e-10, (ratio, mass_error)
                    positive = (
                        state.energy()
                        if method == "graph"
                        else ResidualEnergy(carrier).apply(
                            state.anchor,
                            carrier.project(state.anchor) - state.auxiliary,
                        )
                    )
                    errors = oracle.errors(state.value)
                    result = dict(
                        nfe=n,
                        naux=2 * n,
                        degree=4,
                        steps=steps,
                        end=duration,
                        method=method,
                        scheme=args.scheme,
                        velocity_source=args.velocity_source,
                        wavevectors=WAVES,
                        streamfunction_coefficients=coefficients.tolist(),
                        streamfunction_relative_error=coefficient_error,
                        velocity_relative_probe_error=velocity_error,
                        characteristic_refinement_error=ref_error,
                        sampled_max_cfl=maximum_speed * duration * 2 * n / steps,
                        relative_l2=(cp.asarray(errors["l2"]) / norms).tolist(),
                        relative_fe_floor=floors,
                        energy_ratio=ratio,
                        positive_energy=positive.tolist(),
                        mass_error=mass_error,
                        flow_setup_seconds=setup,
                        evolution_seconds=elapsed,
                        **errors,
                    )
                    output.write(
                        json.dumps(dict(result=result, provenance=meta)) + "\n"
                    )
                    output.flush()
                    print(json.dumps(result), flush=True)
                    if (
                        args.fields
                        and n == max(args.sizes)
                        and steps == max(args.steps)
                    ):
                        fields[method] = state.value.copy()
                if fields:
                    render_fields(
                        args.output.parent / "figures",
                        space,
                        exact,
                        fields,
                        n,
                        steps,
                        args.scheme,
                    )
            del oracle, space, carrier
            cp.get_default_memory_pool().free_all_blocks()


def render_fields(output, space, exact, values, n, steps, scheme):
    from field_report import sample, save
    from report import plt, apply_style
    from matplotlib.colors import LogNorm

    output.mkdir(parents=True, exist_ok=True)
    line = cp.linspace(0, 1, 384, endpoint=False)
    y, x = cp.meshgrid(line, line, indexing="ij")
    points = cp.stack((x, y), axis=-1)
    reference = exact(points)[..., 0]
    methods = list(values)
    sampled = sample(
        space,
        cp.concatenate([values[m][:, :1, :] for m in methods], axis=1),
        points,
        grading=0.0,
    )
    error = cp.abs(sampled - reference[..., None])
    name = f"night-hamiltonian-fields-{scheme}-{n}-{steps}"
    cp.savez_compressed(
        str(output / (name + "-data.npz")),
        reference=reference,
        fields=sampled,
        errors=error,
    )
    reference, error = reference.get(), error.get()
    apply_style()
    fig, axes = plt.subplots(
        1, len(methods) + 1, figsize=(4 * (len(methods) + 1), 4), layout="constrained"
    )
    artist = axes[0].imshow(
        reference, origin="lower", extent=(0, 1, 0, 1), cmap="viridis"
    )
    fig.colorbar(artist, ax=axes[0], shrink=0.7)
    axes[0].set(title="Référence par caractéristiques", xlabel="x", ylabel="y")
    maximum = max(float(error.max()), 1e-300)
    norm = LogNorm(maximum * 1e-6, maximum)
    labels = {
        "reset": "Reconstruction FE",
        "memory": "Mémoire physique",
        "full": "Moments complets",
        "fe-only": "Réaction FE seule",
        "graph": "Réflexion orthogonale",
    }
    for i, method in enumerate(methods):
        artist = axes[i + 1].imshow(
            error[..., i].clip(min=norm.vmin),
            origin="lower",
            extent=(0, 1, 0, 1),
            norm=norm,
            cmap="magma",
        )
        axes[i + 1].set(title="Erreur · " + labels[method], xlabel="x", ylabel="y")
    fig.colorbar(artist, ax=list(axes[1:]), shrink=0.7, label="Erreur absolue")
    save(fig, output, name)
    plt.close(fig)


if __name__ == "__main__":
    main()
