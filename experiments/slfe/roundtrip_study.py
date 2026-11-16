"""GPU deform-and-return tests with repeated changes of auxiliary support.

Distributed under the Boost Software License, Version 1.0.
The prescribed autonomous velocity is reversed after one leg, so the exact
final field equals the initial profile. This tests accumulated transport and
remapping error without a numerical characteristic reference.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Space
from graded_study import GradedMesh
from moving_study import profile_family
from adaptive_study import fe_center
from energy_mesh import from_fe, from_tensor
from carrier import Carrier
from tensor_grid import TensorGrid
from tensor_cellular import TensorCellular
from memory_transport import MemoryTransport
from campaign import provenance


def evolve(
    nfe=128,
    naux=64,
    half_duration=0.25,
    steps=128,
    fields=4,
    method="projected",
    check_every=4,
    sample_order=None,
    energy_floor=0.0,
):
    if method not in ("reset", "projected", "chebyshev"):
        raise ValueError("Unknown memory transfer")
    if check_every < 1:
        raise ValueError("A positive adaptation interval is required")
    space = Space(GradedMesh(nfe, 0.95), 4)
    profile = profile_family(0.01, fields, spread=0.15)
    initial = space.project(profile, degree=20)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    edges, oriented = from_fe(space, initial, naux, energy_floor=energy_floor)
    carrier = Carrier(space, n=naux, edges=edges)
    state = MemoryTransport(carrier, initial)
    center = fe_center(space, initial)
    energy0 = state.energy()
    mass0 = space.mass(initial)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    history = []
    changes = 0
    rebuild = 0.0
    start = time.perf_counter()
    for leg, direction in enumerate((1, -1)):
        flow = TensorCellular(
            carrier,
            direction * half_duration / steps,
            center=center,
            oriented_edges=oriented,
        )
        for first in range(0, steps, check_every):
            count = min(check_every, steps - first)
            for _ in range(count):
                state.advect(flow.apply_carrier)
            next_edges, next_oriented = from_tensor(
                carrier,
                state.auxiliary,
                sample_order=sample_order,
                energy_floor=energy_floor,
            )
            displacement = max(
                float(cp.max(cp.abs(next_edges - carrier.edges))),
                float(cp.max(cp.abs(next_oriented - oriented))),
            )
            loss = None
            change = displacement > 0.01 and not (leg == 1 and first + count == steps)
            if change:
                cp.cuda.Stream.null.synchronize()
                before = time.perf_counter()
                target = Carrier(space, n=naux, edges=next_edges)
                grid = TensorGrid(target, target.edges)
                remap = lambda w: grid.remap(carrier, w)
                if method == "reset":
                    loss = state.restart(target)
                else:
                    loss = state.project_memory(
                        target,
                        remap,
                        iterations=8 if method == "projected" else 4,
                        filter_scale=None if method == "projected" else 0.1,
                    )
                carrier, oriented = target, next_oriented
                flow = TensorCellular(
                    carrier,
                    direction * half_duration / steps,
                    center=center,
                    oriented_edges=oriented,
                )
                changes += 1
                cp.cuda.Stream.null.synchronize()
                rebuild += time.perf_counter() - before
            energy_ratio = float(cp.max(state.energy() / energy0))
            mass_error = float(cp.max(cp.abs(space.mass(state.value) - mass0)))
            assert energy_ratio <= 1 + 3e-8 and mass_error < 3e-8, (
                energy_ratio,
                mass_error,
            )
            history.append(
                dict(
                    leg=leg,
                    step=first + count,
                    changed=change,
                    displacement=displacement,
                    energy_ratio=energy_ratio,
                    mass_error=mass_error,
                    memory_loss=None if loss is None else loss.tolist(),
                )
            )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    errors = space.errors(state.value, profile, 0, degree=20, subdivisions=2)
    floor = space.errors(initial, profile, 0, degree=20, subdivisions=2)["l2"]
    return dict(
        nfe=nfe,
        naux=naux,
        fields=fields,
        half_duration=half_duration,
        steps_per_leg=steps,
        method=method,
        check_every=check_every,
        sample_order=sample_order,
        energy_floor=energy_floor,
        changes=changes,
        setup_seconds=setup,
        evolution_seconds=elapsed,
        rebuild_seconds=rebuild,
        best_fe_l2=floor,
        fe_return_l2=cp.sqrt(space.energy(state.value - initial)).tolist(),
        history=history,
        **errors
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "long"), default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.suite == "smoke":
        cases = [
            dict(nfe=128, naux=64, half_duration=t, steps=128, method=m)
            for t in (0.25, 0.5)
            for m in ("reset", "projected", "chebyshev")
        ]
    else:
        cases = [
            dict(nfe=nfe, naux=naux, half_duration=t, steps=steps, method=m)
            for nfe in (128, 256)
            for naux in (64, 128)
            for t in (0.25, 1.0)
            for steps in (128, 512)
            for m in ("reset", "projected", "chebyshev")
        ]
    meta = provenance()
    with args.output.open("w") as stream:
        for case in cases:
            print(json.dumps(case), flush=True)
            record = evolve(**case)
            stream.write(json.dumps(dict(result=record, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in record.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
