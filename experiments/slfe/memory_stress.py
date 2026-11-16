"""Long mixed-operation CUDA invariant checks for variable-space FE memory.

Distributed under the Boost Software License, Version 1.0.
The local modal rotation is a norm-preserving algebraic stress operation,
not a model of a physical reaction. This suite tests invariants, not PDE error.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Space
from graded_study import GradedMesh
from carrier import Carrier
from tensor_grid import TensorGrid
from adaptive_study import partitions
from memory_transport import MemoryTransport
from campaign import provenance


def run(steps, method):
    space = Space(GradedMesh(8, 0.8), 4)
    centers = cp.asarray([[0.35, 0.45], [0.81, 0.17], [0.03, 0.94]])
    couplings = [
        Carrier(space, n=16, edges=partitions(16, g, centers[i]))
        for i, g in enumerate((0.0, 0.95, 0.9999))
    ]
    grids = [TensorGrid(c, c.edges) for c in couplings]
    remaps = [grids[(i + 1) % 3].remap for i in range(3)]
    flows = []
    for c in couplings:
        y = (c.edges[1, :-1, None] + cp.diff(c.edges[1])[:, None] * c.nodes).ravel()
        flows.append(c.shears([(0, 0.002, cp.sin(2 * cp.pi * y))]))
    initial = cp.random.RandomState(822).normal(size=(space.mesh.cells, 4, space.nb))
    initial[:, :, 0] += 2
    initial /= cp.sqrt(space.energy(initial))[None, :, None]
    state = MemoryTransport(couplings[0], initial)
    energy0 = state.energy()
    mass0 = space.mass(initial)
    angle = cp.linspace(0.01, 0.3, space.mesh.cells)[:, None]
    sine, cosine = cp.sin(angle), cp.cos(angle)

    def physics(v):
        result = v.copy()
        result[:, :, 1] = cosine * v[:, :, 1] - sine * v[:, :, 2]
        result[:, :, 2] = sine * v[:, :, 1] + cosine * v[:, :, 2]
        return result

    maxima = dict(
        energy_growth=0.0,
        observation_error=0.0,
        mass_error=0.0,
        excess_observed_energy=0.0,
        negative_transfer_loss=0.0,
    )
    history = []
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for step in range(steps):
        i, j = step % 3, (step + 1) % 3
        previous_energy = state.energy()
        state.apply_physics(physics)
        state.advect(flows[i].apply_carrier)
        value = state.value.copy()
        remap = lambda w: remaps[i](couplings[i], w)
        if method == "partial":
            loss = state.transfer_memory(couplings[j], remap)
        else:
            loss = state.project_memory(
                couplings[j],
                remap,
                iterations=8 if method == "richardson" else 4,
                filter_scale=None if method == "richardson" else 0.1,
            )
        assert bool(cp.array_equal(value, state.value))
        energy = state.energy()
        observed = state.anchor + couplings[j].adjoint(
            state.auxiliary - couplings[j].project(state.anchor)
        )
        diagnostics = dict(
            energy_growth=float(cp.max((energy - previous_energy) / energy0)),
            observation_error=float(
                cp.max(cp.sqrt(space.energy(observed - state.value) / energy0))
            ),
            mass_error=float(cp.max(cp.abs(space.mass(state.value) - mass0))),
            excess_observed_energy=float(
                cp.max((space.energy(state.value) - energy) / energy0)
            ),
            negative_transfer_loss=float(cp.max(-loss / energy0)),
        )
        for key, value in diagnostics.items():
            maxima[key] = max(maxima[key], value)
        assert max(diagnostics.values()) < 2e-8, (step, diagnostics)
        if step % 100 == 0 or step + 1 == steps:
            history.append(
                dict(
                    step=step + 1,
                    energy_ratio=(energy / energy0).tolist(),
                    **diagnostics
                )
            )
            print(json.dumps(dict(method=method, **history[-1])), flush=True)
    cp.cuda.Stream.null.synchronize()
    return dict(
        steps=steps,
        method=method,
        seconds=time.perf_counter() - start,
        maxima=maxima,
        history=history,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("w") as stream:
        for method in ("partial", "richardson", "chebyshev"):
            result = run(args.steps, method)
            stream.write(
                json.dumps(dict(result=result, provenance=provenance())) + "\n"
            )
            stream.flush()


if __name__ == "__main__":
    main()
