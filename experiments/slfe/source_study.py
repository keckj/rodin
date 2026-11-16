"""GPU transport with a new localized feature injected through the FE solver.

Distributed under the Boost Software License, Version 1.0.
An instantaneous additive FE source is prescribed halfway through the run.
Its energy work is accounted for explicitly; contraction is not claimed across
the forcing event. The reference advects the two analytic profiles separately.
"""

import argparse
import json
import math
from pathlib import Path
import time

from gpu import cp, Mesh, Space
from carrier import Carrier
from tensor_grid import TensorGrid
from tensor_cellular import TensorCellular
from memory_transport import MemoryTransport
from energy_mesh import from_fe, from_tensor
from day_study import reference_feet
from campaign import provenance


def evolve(nfe=128, naux=32, width=0.006, monitor="fe", steps=128, with_fields=False):
    if steps % 8:
        raise ValueError("The midpoint impulse must coincide with a four-step check")
    space = Space(Mesh(nfe, jitter=0.18, seed=817), 4)

    def bump(points, center, sigma):
        delta = cp.sin(cp.pi * (points - cp.asarray(center)))
        return cp.exp(-cp.sum(delta**2, axis=-1) / (2 * (cp.pi * sigma) ** 2)) / (
            math.sqrt(math.pi) * sigma
        )

    initial_profile = lambda points, t: bump(points, (0.35, 0.45), 0.01)
    source_profile = lambda points, t: bump(points, (0.8, 0.75), width)
    end, impulse = 0.25, 0.125

    def exact(points, t):
        return initial_profile(reference_feet(points, -end, 512), 0) + source_profile(
            reference_feet(points, -(end - impulse), 512), 0
        )

    initial = space.project(initial_profile, degree=20)
    source = space.project(source_profile, degree=20)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    edges, oriented = from_fe(space, initial, naux)
    carrier = Carrier(space, n=naux, edges=edges)
    state = MemoryTransport(carrier, initial)
    flow = TensorCellular(carrier, end / steps, oriented_edges=oriented)
    energy_budget = state.energy()
    mass_budget = space.mass(initial)
    history = []
    work_error = None
    missed_source_energy = None
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    for first in range(0, steps, 4):
        count = min(4, steps - first)
        for _ in range(count):
            state.advect(flow.apply_carrier)
        injected = first + count == steps // 2
        if injected:
            old_energy = state.energy()
            work = space.energy(state.value + source) - space.energy(state.value)
            missed_source_energy = (
                space.energy(source) - carrier.energy(carrier.project(source))
            ).tolist()
            state.apply_physics(lambda v: v + source)
            work_error = float(cp.max(cp.abs(state.energy() - old_energy - work)))
            assert work_error < 2e-10, work_error
            energy_budget += work
            mass_budget += space.mass(source)
        if monitor == "fe":
            next_edges, next_oriented = from_fe(space, state.value, naux)
        elif monitor == "auxiliary":
            next_edges, next_oriented = from_tensor(carrier, state.auxiliary)
        else:
            raise ValueError("Expected FE or auxiliary energy monitor")
        motion = max(
            float(cp.max(cp.abs(next_edges - carrier.edges))),
            float(cp.max(cp.abs(next_oriented - oriented))),
        )
        change = first + count < steps and (injected or motion > 0.01)
        loss = None
        if change:
            target = Carrier(space, n=naux, edges=next_edges)
            grid = TensorGrid(target, target.edges)
            loss = state.project_memory(target, lambda w: grid.remap(carrier, w), 8)
            carrier, oriented = target, next_oriented
            flow = TensorCellular(carrier, end / steps, oriented_edges=oriented)
        mass_error = float(cp.max(cp.abs(space.mass(state.value) - mass_budget)))
        energy_excess = float(cp.max(state.energy() - energy_budget))
        assert mass_error < 3e-9 and energy_excess < 3e-9, (mass_error, energy_excess)
        history.append(
            dict(
                step=first + count,
                injected=injected,
                changed=change,
                mass_error=mass_error,
                energy_excess=energy_excess,
                memory_loss=None if loss is None else loss.tolist(),
            )
        )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    errors = space.errors(state.value, exact, 0, degree=20, subdivisions=2)
    floor = space.errors(
        space.project(exact, degree=20), exact, 0, degree=20, subdivisions=2
    )["l2"]
    record = dict(
        nfe=nfe,
        naux=naux,
        width=width,
        monitor=monitor,
        steps=steps,
        end=end,
        impulse_time=impulse,
        setup_seconds=setup,
        evolution_seconds=elapsed,
        best_fe_l2=floor,
        source_work_error=work_error,
        unresolved_source_energy=missed_source_energy,
        history=history,
        **errors
    )
    return (space, initial, state.value, exact, record) if with_fields else record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        dict(nfe=nfe, naux=naux, width=width, monitor=monitor)
        for nfe in ((128,) if args.suite == "smoke" else (128, 256))
        for naux in ((32,) if args.suite == "smoke" else (32, 64))
        for width in ((0.006,) if args.suite == "smoke" else (0.003, 0.006))
        for monitor in ("auxiliary", "fe")
    ]
    meta = provenance()
    with args.output.open("w") as stream:
        for parameters in cases:
            print(json.dumps(parameters), flush=True)
            result = evolve(**parameters)
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
