"""Compare memory transfers on an identical, material-driven support schedule.

Distributed under the Boost Software License, Version 1.0.
Cross-mass matrices and geometric remaps are shared. Three identical-state
transfer timings include diagnostics; common geometric setup is reported apart.
This isolates transfer behavior from solution-dependent mesh feedback.
"""

import argparse
from copy import copy
import json
from pathlib import Path
import statistics
import time

from gpu import cp, Space
from graded_study import GradedMesh
from carrier import Carrier
from tensor_projection import TensorProjection
from tensor_cellular import TensorCellular
from energy_mesh import partitions
from moving_study import profile_family
from day_study import reference_feet
from memory_transport import MemoryTransport
from campaign import provenance


def transfer(state, target, cross, method, gram_backend="einsum"):
    trial = copy(state)
    if method == "reset":
        loss = trial.restart(target)
        diagnostic = dict(energy_loss=loss, ambient_change_squared=loss)
    elif method == "warm":
        diagnostic = trial.project_memory_warm(target, cross, iterations=8)
    elif method.startswith("block"):
        diagnostic = trial.project_memory_galerkin(
            target,
            cross,
            iterations=int(method.removeprefix("block")),
            shared=True,
            gram_backend=gram_backend,
        )
    elif method.startswith("galerkin"):
        diagnostic = trial.project_memory_galerkin(
            target,
            cross,
            iterations=int(method.removeprefix("galerkin")),
            gram_backend=gram_backend,
        )
    else:
        diagnostic = {}
        trial.project_memory(
            target,
            cross,
            iterations=8 if method == "cold" else 4,
            filter_scale=None if method == "cold" else 0.1,
            diagnostics=diagnostic,
        )
    return trial, diagnostic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument(
        "--gram-backend", choices=("einsum", "reduction"), default="einsum"
    )
    args = parser.parse_args()
    meta = provenance()
    methods = (
        tuple(args.methods)
        if args.methods
        else (
            "reset",
            "cold",
            "chebyshev",
            "warm",
            "galerkin0",
            "galerkin1",
            "galerkin2",
            "galerkin4",
        )
    )
    profile = profile_family(0.01, 4, spread=0.15)
    space = Space(GradedMesh(128, 0.95), 4)
    initial = space.project(profile, degree=20)
    cp.cuda.Stream.null.synchronize()
    begin = time.perf_counter()
    coordinate = (cp.arange(512) + 0.5) / 512
    yy, xx = cp.meshgrid(coordinate, coordinate, indexing="ij")
    points = cp.stack((xx, yy), axis=-1)
    weights = cp.sum(profile(points, 0) ** 2, axis=-1) / 512**2
    schedule = [partitions(points, weights, 64)]
    for _ in range(32):
        points = reference_feet(points, 0.25 / 32, 16)
        schedule.append(partitions(points, weights, 64))
    carrier = Carrier(space, n=64, edges=schedule[0][0])
    states = {m: MemoryTransport(carrier, initial) for m in methods}
    energy0, mass0 = space.energy(initial), space.mass(initial)
    cp.cuda.Stream.null.synchronize()
    common_setup = time.perf_counter() - begin
    common_rebuild = 0.0
    history = {m: [] for m in methods}
    advection_seconds = {m: 0.0 for m in methods}
    cursor = 0
    for leg, direction in enumerate((1, -1)):
        flow = TensorCellular(
            carrier, direction * 0.25 / 128, oriented_edges=schedule[cursor][1]
        )
        for check in range(32):

            def operation(w):
                for _ in range(4):
                    w = flow.apply_carrier(w)
                return w

            order = methods[check % len(methods) :] + methods[: check % len(methods)]
            for method in order:
                cp.cuda.Stream.null.synchronize()
                start = time.perf_counter()
                states[method].advect(operation)
                cp.cuda.Stream.null.synchronize()
                advection_seconds[method] += time.perf_counter() - start
            cursor += direction
            if leg == 1 and check == 31:
                continue
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            target = Carrier(space, n=64, edges=schedule[cursor][0])
            cross = TensorProjection(carrier, target)
            flow = TensorCellular(
                target, direction * 0.25 / 128, oriented_edges=schedule[cursor][1]
            )
            cp.cuda.Stream.null.synchronize()
            common_rebuild += time.perf_counter() - start
            for method in order:
                # Warm allocations, then repeat from the exact same input state.
                transfer(
                    states[method],
                    target,
                    cross.apply_carrier,
                    method,
                    args.gram_backend,
                )
                measurements = []
                for _ in range(3):
                    cp.cuda.Stream.null.synchronize()
                    start = time.perf_counter()
                    result, diagnostic = transfer(
                        states[method],
                        target,
                        cross.apply_carrier,
                        method,
                        args.gram_backend,
                    )
                    cp.cuda.Stream.null.synchronize()
                    measurements.append(time.perf_counter() - start)
                states[method] = result
                energy_ratio = float(cp.max(result.energy() / energy0))
                mass_error = float(cp.max(cp.abs(space.mass(result.value) - mass0)))
                assert energy_ratio <= 1 + 3e-8 and mass_error < 3e-8
                history[method].append(
                    dict(
                        leg=leg,
                        step=4 * (check + 1),
                        energy_ratio=energy_ratio,
                        mass_error=mass_error,
                        energy_loss=diagnostic["energy_loss"].tolist(),
                        ambient_change_squared=diagnostic[
                            "ambient_change_squared"
                        ].tolist(),
                        transfer_seconds_median=statistics.median(measurements),
                        transfer_seconds_min=min(measurements),
                        transfer_seconds_max=max(measurements),
                    )
                )
            carrier = target
            if (check + 1) % 8 == 0:
                print(
                    json.dumps(dict(completed_checks=32 * leg + check + 1)), flush=True
                )
    floor = space.errors(initial, profile, 0, degree=20, subdivisions=2)["l2"]
    with args.output.open("w") as stream:
        for method in methods:
            errors = space.errors(
                states[method].value, profile, 0, degree=20, subdivisions=2
            )
            result = dict(
                method=method,
                gram_backend=args.gram_backend,
                nfe=128,
                naux=64,
                fields=4,
                steps_per_leg=128,
                half_duration=0.25,
                changes=63,
                geometry="fixed material schedule",
                common_setup_seconds=common_setup,
                common_rebuild_seconds=common_rebuild,
                advection_seconds=advection_seconds[method],
                transfer_seconds=sum(
                    h["transfer_seconds_median"] for h in history[method]
                ),
                history=history[method],
                best_fe_l2=floor,
                **errors
            )
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )


if __name__ == "__main__":
    main()
