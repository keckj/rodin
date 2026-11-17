"""GPU PDE comparison of three models for spatial reaction of hidden memory.

Distributed under the Boost Software License, Version 1.0.
The prescribed flow and fixed carrier are common. Reference errors are
normalized by the norm of the damped analytical characteristic solution.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Space, launch
from graded_study import GradedMesh
from moving_study import profile_family
from carrier import Carrier
from tensor_cellular import TensorCellular
from memory_transport import MemoryTransport
from reaction_study import Reaction, reference_kernel
from reaction_moments import ReactionMoments
from krylov_study import ErrorOracle
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--steps", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--strengths", nargs="+", type=float, default=[1, 10, 100])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "fe-only",
            "coercive-lift",
            "ambient-moments",
            "increment-moments",
            "increment-reanchor",
            "fe-projected",
            "coercive-projected",
            "reset-projected",
        ),
        default=["fe-only", "coercive-lift", "ambient-moments"],
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--subdivisions", type=int, default=1)
    parser.add_argument("--fields", action="store_true")
    parser.add_argument(
        "--krylov-seed", choices=("moment", "increment", "residual"), default="moment"
    )
    args = parser.parse_args()
    meta = provenance()
    with args.output.open("w") as stream:
        for nfe in args.resolutions:
            space = Space(GradedMesh(nfe, 0.95), 4)
            profile = profile_family(0.01, 1)
            initial = space.project(profile, degree=20)
            carrier = Carrier(space, n=2 * nfe)
            reaction = Reaction(space)
            energy0 = space.energy(initial)
            for strength in args.strengths:

                def exact(points, substeps=512):
                    flattened = cp.ascontiguousarray(points.reshape(-1, 2))
                    traced = cp.empty((len(flattened), 3))
                    launch(
                        reference_kernel(),
                        len(flattened),
                        (flattened, traced, len(flattened), 0.25, substeps),
                    )
                    return (
                        profile(traced[:, :2], 0)
                        * cp.exp(strength * traced[:, 2, None])
                    ).reshape(*points.shape[:-1], 1)

                probe = cp.random.RandomState(13).uniform(size=(4096, 2))
                refined = exact(probe, 1024)
                ref_difference = float(
                    cp.max(cp.abs(exact(probe) - refined)) / cp.max(cp.abs(refined))
                )
                assert ref_difference < 1e-7, ref_difference
                oracle = ErrorOracle(space, profile, 0.25, reference=exact)
                norm = float(
                    cp.sqrt(cp.sum(oracle.reference**2 * oracle.weights[..., None]))
                )
                floor = oracle.errors(oracle.best)["l2"][0] / norm
                for steps in args.steps:
                    print(
                        json.dumps(dict(nfe=nfe, strength=strength, steps=steps)),
                        flush=True,
                    )
                    duration = strength * 0.25 / (2 * steps)
                    flow = TensorCellular(carrier, 0.25 / steps, grading=0.0)
                    local = reaction.exponential(duration)
                    physics = (
                        lambda value, reaction=reaction, local=local: reaction.apply(
                            local, value
                        )
                    )
                    gamma = cp.exp(cp.asarray(-0.5 * duration))
                    cp.cuda.Stream.null.synchronize()
                    start = time.perf_counter()
                    complete = ReactionMoments(
                        carrier, duration, subdivisions=args.subdivisions
                    )
                    cp.cuda.Stream.null.synchronize()
                    complete_setup = time.perf_counter() - start
                    incremental, reanchored, increment_setup, reanchor_setup = (
                        None,
                        None,
                        None,
                        None,
                    )
                    if "increment-moments" in args.methods:
                        start = time.perf_counter()
                        incremental = ReactionMoments(
                            carrier,
                            duration,
                            subdivisions=args.subdivisions,
                            increment=True,
                        )
                        cp.cuda.Stream.null.synchronize()
                        increment_setup = time.perf_counter() - start
                    if "increment-reanchor" in args.methods:
                        start = time.perf_counter()
                        reanchored = ReactionMoments(
                            carrier,
                            duration,
                            subdivisions=args.subdivisions,
                            increment=True,
                            reanchor=True,
                        )
                        cp.cuda.Stream.null.synchronize()
                        reanchor_setup = time.perf_counter() - start
                    field_values = {}
                    for method in args.methods:
                        state = MemoryTransport(carrier, initial)
                        cp.cuda.Stream.null.synchronize()
                        start = time.perf_counter()
                        for _ in range(steps):
                            for half in (0, 1):
                                if method in (
                                    "ambient-moments",
                                    "increment-moments",
                                    "increment-reanchor",
                                ):
                                    operator = (
                                        incremental
                                        if method == "increment-moments"
                                        else (
                                            reanchored
                                            if method == "increment-reanchor"
                                            else complete
                                        )
                                    )
                                    operator.apply(
                                        state,
                                        iterations=args.iterations,
                                        krylov_seed=args.krylov_seed,
                                    )
                                else:
                                    state.apply_physics(
                                        (
                                            complete.apply_fe
                                            if method.endswith("projected")
                                            else physics
                                        ),
                                        hidden_decay=(
                                            0.0
                                            if method == "reset-projected"
                                            else (
                                                gamma
                                                if method.startswith("coercive")
                                                else None
                                            )
                                        ),
                                    )
                                if half == 0:
                                    state.advect(flow.apply_carrier)
                        cp.cuda.Stream.null.synchronize()
                        elapsed = time.perf_counter() - start
                        errors = oracle.errors(state.value)
                        ratio = float(state.energy()[0] / energy0[0])
                        bound_ratio = float(
                            state.energy()[0]
                            / (energy0[0] * cp.exp(cp.asarray(-strength * 0.25)))
                        )
                        assert ratio < 1 + 1e-8
                        if (
                            method.startswith("coercive")
                            or method.endswith("moments")
                            or method == "increment-reanchor"
                            or method == "reset-projected"
                        ):
                            assert bound_ratio <= 1 + 1e-7, bound_ratio
                        result = dict(
                            nfe=nfe,
                            naux=2 * nfe,
                            steps=steps,
                            end=0.25,
                            strength=strength,
                            method=method,
                            iterations=args.iterations,
                            krylov_seed=args.krylov_seed,
                            quadrature_subdivisions=args.subdivisions,
                            relative_l2=errors["l2"][0] / norm,
                            reference_norm=norm,
                            relative_fe_floor=floor,
                            reference_refinement_relative_error=ref_difference,
                            energy_ratio=ratio,
                            coercive_bound_ratio=bound_ratio,
                            weighted_setup_seconds=(
                                reanchor_setup
                                if method == "increment-reanchor"
                                else (
                                    increment_setup
                                    if method == "increment-moments"
                                    else (
                                        complete_setup
                                        if method == "ambient-moments"
                                        else None
                                    )
                                )
                            ),
                            evolution_seconds=elapsed,
                            **errors
                        )
                        stream.write(
                            json.dumps(dict(result=result, provenance=meta)) + "\n"
                        )
                        stream.flush()
                        print(json.dumps(result), flush=True)
                        if args.fields:
                            field_values[method] = state.value.copy()
                    if args.fields:
                        from reaction_fields import render_gpu

                        render_gpu(
                            args.output.parent / "figures",
                            space,
                            exact,
                            field_values,
                            nfe,
                            steps,
                            strength,
                        )
                    del complete, incremental, reanchored
                del oracle
            del carrier, reaction
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
