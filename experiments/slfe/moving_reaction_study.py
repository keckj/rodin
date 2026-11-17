"""Physical reaction and repeated support changes in one GPU PDE calculation.

Distributed under the Boost Software License, Version 1.0.
Uniform, initially adapted and moving auxiliary supports use the same FE
space and number of auxiliary coefficients. Adaptation is checked at fixed
physical intervals. Every method evolves its own monitor; this is a complete
coupled comparison, not a transfer comparison at prescribed geometry.
"""

import argparse
from itertools import product
import json
from pathlib import Path
import time

from gpu import cp, Space, launch
from graded_study import GradedMesh
from moving_study import profile_family
from carrier import Carrier
from tensor_grid import TensorGrid
from tensor_cellular import TensorCellular
from energy_mesh import from_fe, from_tensor
from memory_transport import MemoryTransport
from reaction_study import Reaction, reference_kernel
from reaction_moments import ReactionMoments
from residual_energy import ResidualEnergy
from krylov_study import ErrorOracle
from campaign import provenance


def evolve(
    space,
    initial,
    reaction,
    naux,
    strength,
    steps,
    method,
    geometry,
    transfer="galerkin",
    monitor="auxiliary",
    energy_floor=0.0,
):
    cp.cuda.Stream.null.synchronize()
    begin = time.perf_counter()
    if geometry == "uniform":
        carrier = Carrier(space, n=naux, builder="tiled")
        oriented = None
    else:
        edges, oriented = from_fe(space, initial, naux, energy_floor=energy_floor)
        carrier = Carrier(space, n=naux, edges=edges, builder="tiled")
    dt = 0.25 / steps
    duration = strength * dt / 2
    local = reaction.exponential(duration)
    physics = lambda value: reaction.apply(local, value)
    gamma = float(cp.exp(cp.asarray(-0.5 * duration)))
    complete = (
        ReactionMoments(carrier, duration, builder="tiled")
        if method == "ambient-moments"
        else None
    )
    flow = TensorCellular(carrier, dt, grading=0.0, oriented_edges=oriented)
    state = MemoryTransport(carrier, initial)
    initial_energy = state.energy()
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - begin
    begin = time.perf_counter()
    check_every = max(1, steps // 16)
    changes, rebuild = 0, 0.0
    history = []
    for step in range(steps):
        for half in (0, 1):
            if complete is not None:
                complete.apply(state)
            else:
                state.apply_physics(
                    physics,
                    hidden_decay=(
                        0.0
                        if method == "fe-reset"
                        else gamma if method == "coercive-lift" else None
                    ),
                )
            if half == 0:
                state.advect(flow.apply_carrier)
        if (step + 1) % check_every == 0:
            displacement, changed = 0.0, False
            if geometry == "moving" and step + 1 < steps:
                edges, next_oriented = (
                    from_fe(space, state.value, naux, energy_floor=energy_floor)
                    if monitor == "fe"
                    else from_tensor(
                        carrier, state.auxiliary, energy_floor=energy_floor
                    )
                )
                displacement = max(
                    float(cp.max(cp.abs(edges - carrier.edges))),
                    float(cp.max(cp.abs(next_oriented - oriented))),
                )
                if displacement > 0.01:
                    cp.cuda.Stream.null.synchronize()
                    start = time.perf_counter()
                    target = Carrier(space, n=naux, edges=edges, builder="tiled")
                    grid = TensorGrid(target, target.edges)
                    if transfer == "reset":
                        state.restart(target)
                    elif transfer == "chebyshev":
                        state.project_memory(
                            target,
                            lambda w: grid.remap(carrier, w),
                            iterations=4,
                            filter_scale=0.1,
                        )
                    else:
                        state.project_memory_galerkin(
                            target,
                            lambda w: grid.remap(carrier, w),
                            iterations=2,
                            shared=True,
                            gram_backend="reduction",
                        )
                    carrier, oriented = target, next_oriented
                    if complete is not None:
                        complete = ReactionMoments(carrier, duration, builder="tiled")
                    flow = TensorCellular(
                        carrier, dt, grading=0.0, oriented_edges=oriented
                    )
                    cp.cuda.Stream.null.synchronize()
                    rebuild += time.perf_counter() - start
                    changes += 1
                    changed = True
            ratio = float(state.energy()[0] / initial_energy[0])
            bound = float(cp.exp(cp.asarray(-strength * (step + 1) * dt)))
            assert ratio <= 1 + 1e-8
            if method != "fe-only":
                assert ratio <= bound * (1 + 1e-6) + 1e-28, (ratio, bound)
            history.append(
                dict(
                    step=step + 1,
                    displacement=displacement,
                    changed=changed,
                    energy_ratio=ratio,
                    physical_coercive_bound=bound,
                )
            )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - begin
    residual = carrier.project(state.anchor) - state.auxiliary
    positive_energy = float(ResidualEnergy(carrier).apply(state.anchor, residual)[0])
    visible = state.anchor - carrier.adjoint(residual)
    observation_error = float(cp.sqrt(cp.sum(space.energy(visible - state.value))))
    result = dict(
        nfe=space.mesh.n,
        naux=naux,
        steps=steps,
        strength=strength,
        end=0.25,
        method=method,
        geometry=geometry,
        transfer=transfer,
        monitor=monitor,
        energy_floor=energy_floor,
        check_every=check_every,
        changes=changes,
        setup_seconds=setup,
        evolution_seconds=elapsed,
        rebuild_seconds=rebuild,
        complete_seconds=setup + elapsed,
        positive_energy=positive_energy,
        subtractive_energy=float(state.energy()[0]),
        observation_error=observation_error,
        history=history,
    )
    return state.value, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nfe", type=int, default=128)
    parser.add_argument("--naux", type=int, default=64)
    parser.add_argument("--steps", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--strengths", nargs="+", type=float, default=[1, 100])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("fe-only", "fe-reset", "coercive-lift", "ambient-moments"),
        default=["fe-only", "coercive-lift", "ambient-moments"],
    )
    parser.add_argument(
        "--geometries",
        nargs="+",
        choices=("uniform", "initial", "moving"),
        default=["uniform", "initial", "moving"],
    )
    parser.add_argument(
        "--transfers",
        nargs="+",
        choices=("galerkin", "chebyshev", "reset"),
        default=["galerkin"],
    )
    parser.add_argument(
        "--monitors", nargs="+", choices=("auxiliary", "fe"), default=["auxiliary"]
    )
    parser.add_argument("--energy-floors", nargs="+", type=float, default=[0.0])
    args = parser.parse_args()
    meta = provenance()
    space = Space(GradedMesh(args.nfe, 0.95), 4)
    profile = profile_family(0.01, 1)
    initial = space.project(profile, degree=20)
    reaction = Reaction(space)
    with args.output.open("w") as stream:
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
                    profile(traced[:, :2], 0) * cp.exp(strength * traced[:, 2, None])
                ).reshape(*points.shape[:-1], 1)

            probe = cp.random.RandomState(13).uniform(size=(4096, 2))
            refined = exact(probe, 1024)
            ref_difference = float(
                cp.max(cp.abs(exact(probe) - refined)) / cp.max(cp.abs(refined))
            )
            assert ref_difference < 1e-7
            oracle = ErrorOracle(space, profile, 0.25, reference=exact)
            norm = float(
                cp.sqrt(cp.sum(oracle.reference**2 * oracle.weights[..., None]))
            )
            floor = oracle.errors(oracle.best)["l2"][0] / norm
            for steps, geometry in product(args.steps, args.geometries):
                for method in args.methods:
                    for transfer, monitor, energy_floor in product(
                        args.transfers, args.monitors, args.energy_floors
                    ):
                        print(
                            json.dumps(
                                dict(
                                    strength=strength,
                                    steps=steps,
                                    geometry=geometry,
                                    method=method,
                                )
                            ),
                            flush=True,
                        )
                        value, result = evolve(
                            space,
                            initial,
                            reaction,
                            args.naux,
                            strength,
                            steps,
                            method,
                            geometry,
                            transfer,
                            monitor,
                            energy_floor,
                        )
                        errors = oracle.errors(value)
                        result.update(
                            relative_l2=errors["l2"][0] / norm,
                            reference_norm=norm,
                            relative_fe_floor=floor,
                            reference_refinement_relative_error=ref_difference,
                            **errors
                        )
                        stream.write(
                            json.dumps(dict(result=result, provenance=meta)) + "\n"
                        )
                        stream.flush()
                        print(
                            json.dumps(
                                {k: v for k, v in result.items() if k != "history"}
                            ),
                            flush=True,
                        )
            del oracle


if __name__ == "__main__":
    main()
