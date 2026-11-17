"""GPU transport/reaction of four mass fractions with pointwise sum one.

Distributed under the Boost Software License, Version 1.0.
Shared memory spaces should preserve the linear closure through remapping;
independently chosen nonlinear scalar spaces need not. A cycle-graph reaction
conserves the sum and has an independent characteristic reference.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Space, launch
from graded_study import GradedMesh
from day_study import reference_feet
from energy_mesh import partitions
from carrier import Carrier
from tensor_projection import TensorProjection
from tensor_cellular import TensorCellular
from memory_transport import MemoryTransport
from reaction_study import Reaction, reference_kernel
from krylov_study import ErrorOracle
from frozen_transfer_study import transfer
from system_coupling_study import mix
from campaign import provenance


def evolve(nfe, reactive, output):
    end, steps, naux = 0.25, 128, nfe // 2
    centers = cp.asarray([[0.35, 0.45], [0.40, 0.46], [0.36, 0.51]])
    backgrounds = cp.asarray([0.12, 0.13, 0.15, 0.60])
    amplitudes = cp.asarray([0.12, 0.15, 0.18])

    def profile(points, t):
        delta = cp.sin(cp.pi * (points[..., None, :] - centers))
        bumps = cp.exp(-cp.sum(delta**2, axis=-1) / (2 * (cp.pi * 0.015) ** 2))
        first = backgrounds[:3] + bumps * amplitudes
        return cp.concatenate(
            (first, 1 - cp.sum(first, axis=-1, keepdims=True)), axis=-1
        )

    space = Space(GradedMesh(nfe, 0.95), 4)
    initial = space.project(profile, degree=20)
    constant = cp.zeros((space.mesh.cells, 1, space.nb))
    constant[:, :, 0] = 1
    initial_closure = float(
        cp.sqrt(space.energy(cp.sum(initial, axis=1, keepdims=True) - constant)[0])
    )
    assert initial_closure < 1e-11, initial_closure
    laplacian = cp.asarray(
        [
            [2.0, -1.0, 0.0, -1.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [-1.0, 0.0, -1.0, 2.0],
        ]
    )
    rates, rotation = cp.linalg.eigh(laplacian)
    rates = cp.maximum(rates, 0)
    reaction = Reaction(space) if reactive else None
    matrices = (
        []
        if reaction is None
        else [reaction.exponential(float(rate) * end / (2 * steps)) for rate in rates]
    )

    def physics(value):
        components = mix(value, rotation)
        result = cp.empty_like(components)
        for field, matrix in enumerate(matrices):
            result[:, field : field + 1] = reaction.apply(
                matrix, cp.ascontiguousarray(components[:, field : field + 1])
            )
        return mix(result, rotation.T)

    coordinate = (cp.arange(512) + 0.5) / 512
    yy, xx = cp.meshgrid(coordinate, coordinate, indexing="ij")
    points = cp.stack((xx, yy), axis=-1)
    weights = cp.sum((profile(points, 0) - backgrounds) ** 2, axis=-1) / 512**2
    schedule = [partitions(points, weights, naux)]
    for _ in range(steps // 4):
        points = reference_feet(points, 4 * end / steps, 16)
        schedule.append(partitions(points, weights, naux))
    carrier = Carrier(space, n=naux, edges=schedule[0][0])
    methods = ("chebyshev", "galerkin2", "block2")
    states = {method: MemoryTransport(carrier, initial) for method in methods}
    history = {method: [] for method in methods}
    energy0 = float(cp.sum(space.energy(initial)))
    for check in range(steps // 4):
        flow = TensorCellular(carrier, end / steps, oriented_edges=schedule[check][1])
        for _ in range(4):
            for state in states.values():
                if reactive:
                    state.apply_physics(physics)
                state.advect(flow.apply_carrier)
                if reactive:
                    state.apply_physics(physics)
        if check + 1 < steps // 4:
            target = Carrier(space, n=naux, edges=schedule[check + 1][0])
            cross = TensorProjection(carrier, target)
            for method in methods:
                states[method], _ = transfer(
                    states[method],
                    target,
                    cross.apply_carrier,
                    method,
                    gram_backend="reduction",
                )
            carrier = target
        for method, state in states.items():
            defect = cp.sum(state.value, axis=1, keepdims=True) - constant
            fe_closure = float(cp.sqrt(space.energy(defect)[0]))
            auxiliary_closure = float(
                cp.sqrt(
                    carrier.energy(cp.sum(state.auxiliary, axis=1, keepdims=True) - 1)[
                        0
                    ]
                )
            )
            total_mass_error = float(cp.abs(cp.sum(space.mass(state.value)) - 1))
            ratio = float(cp.sum(state.energy())) / energy0
            assert ratio < 1 + 1e-8 and total_mass_error < 1e-10
            if method != "galerkin2":
                assert max(fe_closure, auxiliary_closure) < 2e-9, (
                    method,
                    fe_closure,
                    auxiliary_closure,
                )
            history[method].append(
                dict(
                    step=4 * (check + 1),
                    fe_closure_l2=fe_closure,
                    auxiliary_closure_l2=auxiliary_closure,
                    total_mass_error=total_mass_error,
                    energy_ratio=ratio,
                )
            )

    def exact(points, substeps=512):
        if not reactive:
            return profile(reference_feet(points, -end, substeps), 0)
        flattened = cp.ascontiguousarray(points.reshape(-1, 2))
        traced = cp.empty((len(flattened), 3))
        launch(
            reference_kernel(),
            len(flattened),
            (flattened, traced, len(flattened), end, substeps),
        )
        values = (profile(traced[:, :2], 0) @ rotation) * cp.exp(
            traced[:, 2, None] * rates
        )
        return (values @ rotation.T).reshape(*points.shape[:-1], 4)

    probe = cp.random.RandomState(13).uniform(size=(4096, 2))
    characteristic_error = float(cp.max(cp.abs(exact(probe) - exact(probe, 1024))))
    assert characteristic_error < 1e-9, characteristic_error
    oracle = ErrorOracle(space, profile, end, reference=exact)
    floor = oracle.errors(oracle.best)["l2"]
    records, maps = [], []
    for method, state in states.items():
        defect = cp.sum(state.value, axis=1) - constant[:, 0]
        maps.append(cp.sqrt(cp.sum(defect**2, axis=1)))
        records.append(
            dict(
                nfe=nfe,
                naux=naux,
                fields=4,
                steps=steps,
                end=end,
                reactive=reactive,
                method=method,
                changes=steps // 4 - 1,
                initial_closure_l2=initial_closure,
                best_fe_l2=floor,
                characteristic_refinement_error=characteristic_error,
                history=history[method],
                **history[method][-1],
                **oracle.errors(state.value),
            )
        )
    # Scientific rendering only: numerical field values and cell RMS were
    # computed on CUDA. Each plotted value belongs to an original FE cell.
    import numpy as np
    from mixture_figures import render

    output.mkdir(exist_ok=True)
    name = f"night-mixture-closure-{nfe}-{int(reactive)}"
    path = output / f"{name}-data.npz"
    np.savez_compressed(
        path,
        triangles=cp.asnumpy(space.mesh.triangles),
        closure_rms=cp.asnumpy(cp.stack(maps)),
    )
    render(path)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    with args.output.open("w") as stream:
        for nfe in (64, 128):
            for reactive in (False, True):
                print(json.dumps(dict(nfe=nfe, reactive=reactive)), flush=True)
                for result in evolve(nfe, reactive, args.output.parent / "figures"):
                    stream.write(
                        json.dumps(dict(result=result, provenance=meta)) + "\n"
                    )
                    stream.flush()
                    print(
                        json.dumps({k: v for k, v in result.items() if k != "history"}),
                        flush=True,
                    )
                cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
