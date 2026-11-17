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
    monitor_source="auxiliary",
    floating_phase=False,
    phase_metric="probability",
    remap_budget=None,
    budget_allocation="uniform",
    gram_backend="einsum",
    carrier_builder="auto",
    krylov_seed="moment",
):
    if method not in ("reset", "projected", "chebyshev", "warm", "galerkin", "block"):
        raise ValueError("Unknown memory transfer")
    if check_every < 1:
        raise ValueError("A positive adaptation interval is required")
    if monitor_source not in ("auxiliary", "fe"):
        raise ValueError("Expected an auxiliary or FE energy monitor")
    if remap_budget is not None and (remap_budget <= 0 or method != "galerkin"):
        raise ValueError("A positive remapping budget requires the Galerkin transfer")
    if budget_allocation not in ("uniform", "remaining"):
        raise ValueError("Expected uniform or remaining-budget allocation")
    if gram_backend not in ("einsum", "reduction"):
        raise ValueError("Expected einsum or reduction Gram assembly")
    if remap_budget is not None and carrier_builder != "auto":
        raise ValueError("The budget controller currently uses the default builder")
    possible_changes = 2 * ((steps + check_every - 1) // check_every) - 1
    budget_used = 0.0
    space = Space(GradedMesh(nfe, 0.95), 4)
    profile = profile_family(0.01, fields, spread=0.15)
    initial = space.project(profile, degree=20)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    edges, oriented = from_fe(space, initial, naux, energy_floor=energy_floor)
    carrier = Carrier(space, n=naux, edges=edges, builder=carrier_builder)
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
            previous = (
                cp.concatenate((carrier.edges, oriented)) if floating_phase else None
            )
            if monitor_source == "fe":
                next_edges, next_oriented = from_fe(
                    space,
                    state.value,
                    naux,
                    energy_floor=energy_floor,
                    previous_edges=previous,
                    phase_metric=phase_metric,
                )
            else:
                next_edges, next_oriented = from_tensor(
                    carrier,
                    state.auxiliary,
                    sample_order=sample_order,
                    energy_floor=energy_floor,
                    previous_edges=previous,
                    phase_metric=phase_metric,
                )
            displacement = max(
                float(cp.max(cp.abs(next_edges - carrier.edges))),
                float(cp.max(cp.abs(next_oriented - oriented))),
            )
            loss = None
            attempts = []
            fraction = None
            change = displacement > 0.01 and not (leg == 1 and first + count == steps)
            if change:
                cp.cuda.Stream.null.synchronize()
                before = time.perf_counter()
                if remap_budget is not None:
                    from remap_controller import budgeted_move

                    state, fraction, diagnostic, attempts = budgeted_move(
                        state,
                        next_edges,
                        (
                            remap_budget / possible_changes
                            if budget_allocation == "uniform"
                            else max(remap_budget - budget_used, 1e-300)
                        ),
                    )
                    change = fraction > 0
                    target = state.coupling
                    next_oriented = oriented + fraction * (next_oriented - oriented)
                    if change:
                        loss = diagnostic["energy_loss"]
                        budget_used += attempts[-1]["measured_distance"]
                        assert budget_used <= remap_budget * (1 + 1e-12)
                else:
                    target = Carrier(
                        space, n=naux, edges=next_edges, builder=carrier_builder
                    )
                    grid = TensorGrid(target, target.edges)
                    remap = lambda w: grid.remap(carrier, w)
                    if method == "reset":
                        loss = state.restart(target)
                    elif method == "warm":
                        diagnostic = state.project_memory_warm(
                            target, remap, iterations=8
                        )
                        loss = diagnostic["energy_loss"]
                    elif method in ("galerkin", "block"):
                        diagnostic = state.project_memory_galerkin(
                            target,
                            remap,
                            iterations=2,
                            shared=method == "block",
                            gram_backend=gram_backend,
                            krylov_seed=krylov_seed,
                        )
                        loss = diagnostic["energy_loss"]
                    else:
                        loss = state.project_memory(
                            target,
                            remap,
                            iterations=8 if method == "projected" else 4,
                            filter_scale=None if method == "projected" else 0.1,
                        )
                if change:
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
                    transfer_fraction=fraction,
                    transfer_attempts=attempts,
                    remap_budget_used=budget_used,
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
        gram_backend=gram_backend,
        carrier_builder=carrier_builder,
        krylov_seed=krylov_seed,
        check_every=check_every,
        sample_order=sample_order,
        energy_floor=energy_floor,
        monitor_source=monitor_source,
        floating_phase=floating_phase,
        phase_metric=phase_metric,
        remap_budget=remap_budget,
        budget_allocation=budget_allocation,
        remap_budget_used=budget_used,
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
    parser.add_argument("--resume", action="store_true")
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
    keys = ("nfe", "naux", "half_duration", "steps_per_leg", "method")
    completed = set()
    if args.resume and args.output.exists():
        completed = {
            tuple(json.loads(line)["result"][k] for k in keys)
            for line in args.output.read_text().splitlines()
        }
    with args.output.open("a" if args.resume else "w") as stream:
        for case in cases:
            key = (
                case["nfe"],
                case["naux"],
                case["half_duration"],
                case["steps"],
                case["method"],
            )
            if key in completed:
                continue
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
