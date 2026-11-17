"""GPU advection of a reacting four-species mixture near a fixed equilibrium.

Distributed under the Boost Software License, Version 1.0.
We evolve deviations from Y_eq=(1/4,...,1/4), so the analytical reference
and FE floor remain meaningful when reaction nearly reaches equilibrium.
The cycle graph conserves the sum. No positivity limiter is applied.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Space, launch
from graded_study import GradedMesh
from carrier import Carrier
from memory_transport import MemoryTransport
from tensor_cellular import TensorCellular
from reaction_study import Reaction, reference_kernel
from spectral_reaction_moments import SpectralReactionMoments
from spectral_reaction_validation import cycle_matrix
from system_coupling_study import mix
from krylov_study import ErrorOracle
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--steps", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--strengths", nargs="+", type=float, default=[1, 10, 20])
    args = parser.parse_args()
    meta = provenance()
    centers = cp.asarray([[0.35, 0.45], [0.40, 0.46], [0.36, 0.51]])
    amplitudes = cp.asarray([0.04, 0.05, 0.06])

    def deviation(points, t):
        delta = cp.sin(cp.pi * (points[..., None, :] - centers))
        first = (
            cp.exp(-cp.sum(delta**2, axis=-1) / (2 * (cp.pi * 0.015) ** 2)) * amplitudes
        )
        return cp.concatenate((first, -cp.sum(first, axis=-1, keepdims=True)), axis=-1)

    matrix = cycle_matrix()
    rates, rotation = cp.linalg.eigh(matrix)
    rates = cp.where(rates > 1e-12, rates, 0)
    with args.output.open("w") as stream:
        for nfe in args.resolutions:
            space = Space(GradedMesh(nfe, 0.95), 4)
            initial = space.project(deviation, degree=20)
            initial[:, -1] = -cp.sum(initial[:, :-1], axis=1)
            carrier = Carrier(space, n=2 * nfe)
            reaction = Reaction(space)
            initial_energy = float(cp.sum(space.energy(initial)))
            for strength in args.strengths:

                def exact(points, substeps=512):
                    flat = cp.ascontiguousarray(points.reshape(-1, 2))
                    traced = cp.empty((len(flat), 3))
                    launch(
                        reference_kernel(),
                        len(flat),
                        (flat, traced, len(flat), 0.25, substeps),
                    )
                    modes = (deviation(traced[:, :2], 0) @ rotation) * cp.exp(
                        strength * traced[:, 2, None] * rates
                    )
                    return (modes @ rotation.T).reshape(*points.shape[:-1], 4)

                probe = cp.random.RandomState(13).uniform(size=(4096, 2))
                reference_difference = float(
                    cp.max(cp.abs(exact(probe) - exact(probe, 1024)))
                    / cp.max(cp.abs(exact(probe, 1024)))
                )
                assert reference_difference < 1e-7, reference_difference
                oracle = ErrorOracle(space, deviation, 0.25, reference=exact)
                norm = float(
                    cp.sqrt(cp.sum(oracle.reference**2 * oracle.weights[..., None]))
                )
                floor = (
                    float(
                        cp.sqrt(
                            cp.sum(cp.asarray(oracle.errors(oracle.best)["l2"]) ** 2)
                        )
                    )
                    / norm
                )
                for steps in args.steps:
                    print(
                        json.dumps(dict(nfe=nfe, strength=strength, steps=steps)),
                        flush=True,
                    )
                    duration = strength * 0.25 / (2 * steps)
                    flow = TensorCellular(carrier, 0.25 / steps, grading=0.0)
                    matrices = [
                        None if rate == 0 else reaction.exponential(duration * rate)
                        for rate in rates.tolist()
                    ]
                    decay = cp.exp(-0.5 * duration * rates)
                    damping = cp.ascontiguousarray(
                        (rotation * decay[None, :]) @ rotation.T
                    )

                    def physics(value, reaction=reaction, matrices=matrices):
                        modes = mix(value, rotation)
                        output = cp.empty_like(modes)
                        for field, local in enumerate(matrices):
                            part = cp.ascontiguousarray(modes[:, field : field + 1])
                            output[:, field : field + 1] = (
                                part if local is None else reaction.apply(local, part)
                            )
                        return mix(output, rotation.T)

                    cp.cuda.Stream.null.synchronize()
                    start = time.perf_counter()
                    complete = SpectralReactionMoments(carrier, matrix, duration)
                    cp.cuda.Stream.null.synchronize()
                    setup = time.perf_counter() - start
                    for method in ("fe-only", "modal-lift", "ambient-moments"):
                        state = MemoryTransport(carrier, initial)
                        cp.cuda.Stream.null.synchronize()
                        start = time.perf_counter()
                        for _ in range(steps):
                            for half in (0, 1):
                                if method == "ambient-moments":
                                    complete.apply(state)
                                elif method == "fe-only":
                                    state.apply_physics(physics)
                                else:
                                    new = physics(state.value)
                                    delta = new - mix(state.value, damping)
                                    state.anchor = mix(state.anchor, damping) + delta
                                    state.auxiliary = mix(
                                        state.auxiliary, damping
                                    ) + carrier.project(delta)
                                    state.value = new
                                if half == 0:
                                    state.advect(flow.apply_carrier)
                        cp.cuda.Stream.null.synchronize()
                        elapsed = time.perf_counter() - start
                        errors = oracle.errors(state.value)
                        relative = (
                            float(cp.sqrt(cp.sum(cp.asarray(errors["l2"]) ** 2))) / norm
                        )
                        closure = float(
                            cp.sqrt(
                                space.energy(
                                    cp.sum(state.value, axis=1, keepdims=True)
                                )[0]
                            )
                        )
                        auxiliary_closure = float(
                            cp.sqrt(
                                carrier.energy(
                                    cp.sum(state.auxiliary, axis=1, keepdims=True)
                                )[0]
                            )
                        )
                        ratio = float(cp.sum(state.energy())) / initial_energy
                        assert (
                            ratio <= 1 + 1e-8 and max(closure, auxiliary_closure) < 2e-9
                        ), (ratio, closure, auxiliary_closure)
                        result = dict(
                            nfe=nfe,
                            naux=2 * nfe,
                            fields=4,
                            steps=steps,
                            strength=strength,
                            end=0.25,
                            method=method,
                            relative_l2=relative,
                            relative_fe_floor=floor,
                            reference_deviation_norm=norm,
                            reference_refinement_relative_error=reference_difference,
                            fe_closure_l2=closure,
                            auxiliary_closure_l2=auxiliary_closure,
                            energy_ratio=ratio,
                            weighted_setup_seconds=(
                                setup if method == "ambient-moments" else None
                            ),
                            evolution_seconds=elapsed,
                            minimum_fraction_sampled=0.25 + min(errors["min_sampled"]),
                            **errors
                        )
                        stream.write(
                            json.dumps(dict(result=result, provenance=meta)) + "\n"
                        )
                        stream.flush()
                        print(json.dumps(result), flush=True)
                    del complete
                del oracle
            del carrier, reaction, initial
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
