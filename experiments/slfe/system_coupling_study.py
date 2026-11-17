"""Coordinate-covariance of GPU FE transport with a four-component reaction.

Distributed under the Boost Software License, Version 1.0.
The physical reaction is diagonal in one coordinate system and dense in a
rotated one. All methods follow the same prescribed material support schedule.
Independent characteristic integration supplies the PDE reference.
"""

import argparse
from copy import copy
import json
from pathlib import Path
import time

from gpu import cp, Space, launch
from graded_study import GradedMesh
from moving_study import profile_family
from day_study import reference_feet
from energy_mesh import partitions
from carrier import Carrier
from tensor_projection import TensorProjection
from tensor_cellular import TensorCellular
from memory_transport import MemoryTransport
from reaction_study import Reaction, reference_kernel
from krylov_study import ErrorOracle
from frozen_transfer_study import transfer
from campaign import provenance


def mix(value, matrix):
    return cp.ascontiguousarray(cp.einsum("cfb,fg->cgb", value, matrix))


def evolve(nfe, spread):
    end, steps, naux = 0.25, 128, 64
    profile = profile_family(0.01, 4, spread=spread)
    space = Space(GradedMesh(nfe, 0.95), 4)
    initial = space.project(profile, degree=20)
    rotation, _ = cp.linalg.qr(cp.random.RandomState(91).normal(size=(4, 4)))
    rates = cp.asarray([0.5, 1.0, 2.0, 4.0])
    reaction = Reaction(space)
    matrices = [reaction.exponential(float(rate) * end / (2 * steps)) for rate in rates]

    def physics(value, rotated):
        if rotated:
            value = mix(value, rotation.T)
        result = cp.empty_like(value)
        for field, matrix in enumerate(matrices):
            result[:, field : field + 1] = reaction.apply(
                matrix, cp.ascontiguousarray(value[:, field : field + 1])
            )
        return mix(result, rotation) if rotated else result

    coordinate = (cp.arange(512) + 0.5) / 512
    yy, xx = cp.meshgrid(coordinate, coordinate, indexing="ij")
    points = cp.stack((xx, yy), axis=-1)
    weights = cp.sum(profile(points, 0) ** 2, axis=-1) / 512**2
    schedule = [partitions(points, weights, naux)]
    for _ in range(steps // 4):
        points = reference_feet(points, 4 * end / steps, 16)
        schedule.append(partitions(points, weights, naux))
    carrier = Carrier(space, n=naux, edges=schedule[0][0])
    methods = ("chebyshev", "galerkin2", "block2")
    states = {
        (method, rotated): MemoryTransport(
            carrier, mix(initial, rotation) if rotated else initial
        )
        for method in methods
        for rotated in (False, True)
    }
    energy0 = float(cp.sum(space.energy(initial)))
    history = []
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for check in range(steps // 4):
        flow = TensorCellular(carrier, end / steps, oriented_edges=schedule[check][1])
        for _ in range(4):
            for (method, rotated), state in states.items():
                operation = lambda value, rotated=rotated: physics(value, rotated)
                state.apply_physics(operation)
                state.advect(flow.apply_carrier)
                state.apply_physics(operation)
        if check + 1 < steps // 4:
            target = Carrier(space, n=naux, edges=schedule[check + 1][0])
            cross = TensorProjection(carrier, target)
            for key in states:
                states[key], _ = transfer(
                    states[key],
                    target,
                    cross.apply_carrier,
                    key[0],
                    gram_backend="reduction",
                )
            carrier = target
        for method in methods:
            original, rotated = states[(method, False)], states[(method, True)]
            difference = original.value - mix(rotated.value, rotation.T)
            discrepancy = float(cp.sqrt(cp.sum(space.energy(difference))))
            ratio = max(
                float(cp.sum(state.energy())) / energy0 for state in (original, rotated)
            )
            assert ratio <= 1 + 1e-8, ratio
            if method != "galerkin2":
                assert discrepancy < 2e-9, dict(
                    method=method, discrepancy=discrepancy, check=check
                )
            history.append(
                dict(
                    step=4 * (check + 1),
                    method=method,
                    fe_covariance_l2=discrepancy,
                    energy_ratio=ratio,
                )
            )
    cp.cuda.Stream.null.synchronize()
    common_evolution_seconds = time.perf_counter() - start

    def exact(points, substeps=512):
        flattened = cp.ascontiguousarray(points.reshape(-1, 2))
        traced = cp.empty((len(flattened), 3))
        launch(
            reference_kernel(),
            len(flattened),
            (flattened, traced, len(flattened), end, substeps),
        )
        return (profile(traced[:, :2], 0) * cp.exp(traced[:, 2, None] * rates)).reshape(
            *points.shape[:-1], 4
        )

    probe = cp.random.RandomState(13).uniform(size=(4096, 2))
    characteristic_error = float(cp.max(cp.abs(exact(probe) - exact(probe, 1024))))
    assert characteristic_error < 1e-8, characteristic_error
    oracle = ErrorOracle(space, profile, end, reference=exact)
    floor = oracle.errors(oracle.best)["l2"]
    records = []
    for method in methods:
        original, rotated = states[(method, False)], states[(method, True)]
        unrotated = mix(rotated.value, rotation.T)
        difference = original.value - unrotated
        delta = copy(original)
        delta.anchor = original.anchor - mix(rotated.anchor, rotation.T)
        delta.auxiliary = original.auxiliary - mix(rotated.auxiliary, rotation.T)
        records.append(
            dict(
                nfe=nfe,
                naux=naux,
                fields=4,
                spread=spread,
                steps=steps,
                end=end,
                method=method,
                changes=steps // 4 - 1,
                gram_backend="reduction",
                rates=rates.tolist(),
                rotation=rotation.tolist(),
                best_fe_l2=floor,
                fe_covariance_l2=float(cp.sqrt(cp.sum(space.energy(difference)))),
                ambient_covariance_l2=float(cp.sqrt(cp.sum(delta.energy()).clip(0))),
                characteristic_refinement_error=characteristic_error,
                diagonal_basis_errors=oracle.errors(original.value),
                rotated_basis_errors=oracle.errors(unrotated),
                history=[x for x in history if x["method"] == method],
                common_evolution_seconds=common_evolution_seconds,
            )
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    with args.output.open("w") as stream:
        for nfe in (64, 128):
            for spread in (0.008, 0.15):
                print(json.dumps(dict(nfe=nfe, spread=spread)), flush=True)
                for result in evolve(nfe, spread):
                    stream.write(
                        json.dumps(dict(result=result, provenance=meta)) + "\n"
                    )
                    stream.flush()
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in result.items()
                                if k not in ("history", "rotation")
                            }
                        ),
                        flush=True,
                    )
                cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
