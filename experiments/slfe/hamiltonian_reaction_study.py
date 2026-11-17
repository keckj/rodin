"""GPU physical reaction between conservative Hamiltonian transport stages.

Distributed under the Boost Software License, Version 1.0.
The outer reaction/transport splitting is second order with positive
reaction durations. Only the interior advection uses signed Yoshida stages.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, HERE, Mesh, Space, launch
from carrier import Carrier
from tensor_hamiltonian import ComposedHamiltonian, streamfunction_modes
from hamiltonian_study import WAVES, profile, render_fields
from memory_transport import MemoryTransport
from reaction_moments import ReactionMoments
from residual_energy import ResidualEnergy
from krylov_study import ErrorOracle
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--steps", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--strengths", type=float, nargs="+", default=[1, 100])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("full", "target", "reset", "fe-only"),
        default=["full", "reset", "fe-only"],
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--subdivisions", type=int, default=1)
    parser.add_argument("--fields", action="store_true")
    args = parser.parse_args()
    meta = provenance()
    duration = 0.25
    waves = cp.asarray(WAVES, dtype=cp.float64)
    denominator = cp.sum(waves**2, axis=1)
    analytic = cp.stack(
        (0.018 / denominator, 0.009 * (-1.0) ** cp.arange(len(WAVES)) / denominator),
        axis=1,
    )
    modes = cp.ascontiguousarray(cp.concatenate((waves, analytic), axis=1))
    source = (HERE / "hamiltonian_reference.cu").read_text() + (
        HERE / "hamiltonian_reaction_reference.cu"
    ).read_text()
    kernel = cp.RawKernel(source, "hamiltonian_reaction_reference")
    kernel.compile()

    def traced(points, steps=512, coefficients=modes):
        flat = cp.ascontiguousarray(points.reshape(-1, 2))
        result = cp.empty((len(flat), 3))
        launch(
            kernel,
            len(flat),
            (flat, result, len(flat), coefficients, len(WAVES), duration, steps),
        )
        return result.reshape(*points.shape[:-1], 3)

    def psi(points, time):
        angle = 2 * cp.pi * cp.einsum("...i,mi->...m", points, waves)
        return cp.sum(
            cp.cos(angle) * analytic[:, 0] + cp.sin(angle) * analytic[:, 1], axis=-1
        )[..., None]

    probe = cp.random.RandomState(612).uniform(size=(4096, 2))
    zero = modes.copy()
    zero[:, 2:] = 0
    static = traced(probe, coefficients=zero)
    static_error = float(cp.max(cp.abs(static[:, :2] - probe)))
    static_rate_error = float(
        cp.max(
            cp.abs(
                static[:, 2] + duration * (1 + 0.5 * cp.sin(2 * cp.pi * probe[:, 0]))
            )
        )
    )
    assert max(static_error, static_rate_error) < 1e-12
    with args.output.open("w") as output:
        for n in args.sizes:
            space = Space(Mesh(n, jitter=0.18, seed=421), 4)
            initial = space.project(profile, degree=20)
            coefficients = streamfunction_modes(
                space, space.project(psi, degree=20), WAVES
            )
            carrier = Carrier(space, n=2 * n, builder="tiled")
            initial_energy = space.energy(initial)
            for strength in args.strengths:

                def exact(points, substeps=512):
                    values = traced(points, substeps)
                    return profile(values[..., :2], 0) * cp.exp(
                        strength * values[..., 2, None]
                    )

                refined = exact(probe, 1024)
                ref_error = float(
                    cp.max(cp.abs(exact(probe) - refined)) / cp.max(cp.abs(refined))
                )
                assert ref_error < 1e-9, ref_error
                oracle = ErrorOracle(space, profile, duration, reference=exact)
                norms = cp.sqrt(
                    cp.sum(oracle.reference**2 * oracle.weights[..., None], axis=(0, 1))
                )
                floor = (cp.asarray(oracle.errors(oracle.best)["l2"]) / norms).tolist()
                for steps in args.steps:
                    cp.cuda.Stream.null.synchronize()
                    begin = time.perf_counter()
                    flow = ComposedHamiltonian(
                        carrier, WAVES, coefficients, duration / steps, "yoshida6"
                    )
                    flow.apply_carrier(carrier.project(initial))
                    cp.cuda.Stream.null.synchronize()
                    flow_setup = time.perf_counter() - begin
                    begin = time.perf_counter()
                    reaction = ReactionMoments(
                        carrier,
                        strength * duration / (2 * steps),
                        builder="tiled",
                        subdivisions=args.subdivisions,
                    )
                    cp.cuda.Stream.null.synchronize()
                    reaction_setup = time.perf_counter() - begin
                    fields = {}
                    for method in args.methods:
                        state = MemoryTransport(carrier, initial)
                        cp.cuda.Stream.null.synchronize()
                        begin = time.perf_counter()
                        for _ in range(steps):
                            for half in (0, 1):
                                if method in ("full", "target"):
                                    reaction.apply(
                                        state,
                                        iterations=(
                                            1 if method == "target" else args.iterations
                                        ),
                                        target_seed=method == "target",
                                    )
                                else:
                                    state.apply_physics(
                                        reaction.apply_fe,
                                        hidden_decay=0.0 if method == "reset" else None,
                                    )
                                if half == 0:
                                    state.advect(flow.apply_carrier)
                        cp.cuda.Stream.null.synchronize()
                        elapsed = time.perf_counter() - begin
                        energy = ResidualEnergy(carrier).apply(
                            state.anchor,
                            carrier.project(state.anchor) - state.auxiliary,
                        )
                        bound = float(
                            cp.max(
                                energy
                                / (
                                    initial_energy
                                    * cp.exp(cp.asarray(-strength * duration))
                                )
                            )
                        )
                        if method != "fe-only":
                            assert bound <= 1 + 1e-8, bound
                        errors = oracle.errors(state.value)
                        result = dict(
                            nfe=n,
                            naux=2 * n,
                            degree=4,
                            end=duration,
                            steps=steps,
                            strength=strength,
                            method=method,
                            iterations=args.iterations,
                            subdivisions=args.subdivisions,
                            scheme="positive-Strang-reaction/Yoshida6-advection",
                            relative_l2=(cp.asarray(errors["l2"]) / norms).tolist(),
                            relative_fe_floor=floor,
                            reference_norms=norms.tolist(),
                            reference_refinement_relative_error=ref_error,
                            reference_static_error=static_error,
                            reference_static_rate_error=static_rate_error,
                            positive_energy=energy.tolist(),
                            coercive_bound_ratio=bound,
                            flow_setup_seconds=flow_setup,
                            weighted_setup_seconds=reaction_setup,
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
                            and steps == max(args.steps)
                            and strength == max(args.strengths)
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
                            f"reaction{strength:g}-yoshida6",
                        )
                    del flow, reaction, state
                    cp.get_default_memory_pool().free_all_blocks()
                del oracle


if __name__ == "__main__":
    main()
