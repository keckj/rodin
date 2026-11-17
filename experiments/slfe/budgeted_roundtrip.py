"""GPU experiment with an explicit measured-memory-transfer budget.

Distributed under the Boost Software License, Version 1.0.
Mesh motion is backtracked, while the proposed monitor and transport stay fixed.
The allocated budget bounds remapping defects along the realized operator path;
it does not bound the full PDE error or compare different controller paths.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from carrier import Carrier
from tensor_grid import TensorGrid
from memory_transport import MemoryTransport
from remap_controller import budgeted_move
from roundtrip_study import evolve
from campaign import provenance


def validate():
    space = Space(Mesh(8, jitter=0.18, seed=817), 4)
    old = Carrier(space, n=8)
    initial = cp.random.RandomState(183).normal(size=(space.mesh.cells, 2, space.nb))
    state = MemoryTransport(old, initial)
    state.advect(old.transport(0.07, "shear").apply_carrier)
    snapshots = [x.copy() for x in (state.value, state.anchor, state.auxiliary)]
    t = cp.linspace(0, 1, 9)
    edges = cp.stack((t + 0.01 * cp.sin(2 * cp.pi * t), t))
    records = []
    for tolerance in (1.0, 0.001, 1e-8):
        result, fraction, _, attempts = budgeted_move(
            state, edges, tolerance, maximum_backtracks=2
        )
        assert all(
            bool(cp.array_equal(a, b))
            for a, b in zip(snapshots, (state.value, state.anchor, state.auxiliary))
        )
        if fraction == 0:
            assert result is state
            distance_squared = 0.0
        else:
            assert attempts[-1]["measured_distance"] <= tolerance
            new = result.coupling
            d = state.auxiliary - old.project(state.anchor)
            e = result.auxiliary - new.project(result.anchor)
            mass_old = cp.broadcast_to(old.mass, (old.n**2, old.nc))[:, None, :]
            mass_new = cp.broadcast_to(new.mass, (new.n**2, new.nc))[:, None, :]
            cross = TensorGrid(new, new.edges).remap(old, d)
            inner = cp.sum(
                state.anchor * result.anchor * space.mesh.areas[:, None, None],
                axis=(0, 2),
            )
            inner += cp.sum(d * old.project(result.anchor) * mass_old, axis=(0, 2))
            inner += cp.sum(e * new.project(state.anchor) * mass_new, axis=(0, 2))
            inner += cp.sum(e * cross * mass_new, axis=(0, 2))
            distance_squared = float(
                cp.max(state.energy() + result.energy() - 2 * inner)
            )
            assert distance_squared <= tolerance**2 + 1e-10
        records.append(
            dict(
                tolerance=tolerance,
                fraction=fraction,
                distance_squared=distance_squared,
                attempts=attempts,
            )
        )
    assert records[0]["fraction"] > 0 and records[-1]["fraction"] == 0
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allocation", choices=("uniform", "remaining"), default="uniform"
    )
    args = parser.parse_args()
    cases = [
        dict(remap_budget=b, steps=128, check_every=4) for b in (0.003, 0.01, 0.03)
    ]
    cases += [dict(remap_budget=0.01, steps=512, check_every=16)]
    if args.allocation == "remaining":
        cases = [dict(remap_budget=b, steps=128, check_every=4) for b in (0.01, 0.03)]
    meta = provenance()
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(dict(results=validate(), provenance=meta), indent=2) + "\n"
    )
    with args.output.open("w") as stream:
        for parameters in cases:
            print(json.dumps(parameters), flush=True)
            result = evolve(
                method="galerkin", budget_allocation=args.allocation, **parameters
            )
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
