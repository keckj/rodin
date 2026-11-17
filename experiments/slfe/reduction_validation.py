"""GPU weighted-product oracle and unchanged-numerics transfer comparisons.

Distributed under the Boost Software License, Version 1.0.
"""

import argparse
from copy import copy
import json
from pathlib import Path
import statistics
import time

from gpu import cp, Mesh, Space
from carrier import Carrier
from tensor_grid import TensorGrid
from memory_transport import MemoryTransport
from weighted_products import weighted_products
from campaign import provenance


def validate():
    products = []
    rng = cp.random.RandomState(417)
    for points in (1, 17, 1025, 8193):
        left = rng.normal(size=(4, points, 6))[..., ::2]
        right = rng.normal(size=(4, points, 5))
        weights = rng.uniform(size=points)
        reference = cp.einsum("bik,i,bil->bkl", left, weights, right)
        actual = weighted_products(left, right, weights)
        error = float(cp.max(cp.abs(actual - reference)))
        relative = error / max(1.0, float(cp.max(cp.abs(reference))))
        assert relative < 1e-12, relative
        products.append(
            dict(points=points, absolute_error=error, relative_error=relative)
        )
    transfers = []
    for degree in (1, 2, 4):
        space = Space(Mesh(8, jitter=0.18, seed=817), degree)
        carrier = Carrier(space, n=8)
        initial = rng.normal(size=(space.mesh.cells, 4, space.nb))
        original = MemoryTransport(carrier, initial)
        original.advect(carrier.transport(0.07, "shear").apply_carrier)
        original.apply_physics(lambda v: 0.9 * v)
        original.advect(carrier.transport(-0.04, "translation").apply_carrier)
        for motion in (0.0, 0.01):
            coordinate = cp.linspace(0, 1, carrier.n + 1)
            edges = cp.stack(
                (
                    coordinate + motion * cp.sin(2 * cp.pi * coordinate) / (2 * cp.pi),
                    coordinate,
                )
            )
            target = Carrier(space, n=8, edges=edges)
            grid = TensorGrid(target, target.edges)
            cross = lambda w: grid.remap(carrier, w)
            for iterations in (0, 2, 4):
                for shared in (False, True):
                    reference, actual = copy(original), copy(original)
                    reference.project_memory_galerkin(
                        target, cross, iterations, shared, "einsum"
                    )
                    actual.project_memory_galerkin(
                        target, cross, iterations, shared, "reduction"
                    )
                    coefficient_error = max(
                        float(
                            cp.max(
                                cp.abs(getattr(actual, key) - getattr(reference, key))
                            )
                        )
                        for key in ("anchor", "auxiliary", "value")
                    )
                    energy_error = float(
                        cp.max(cp.abs(actual.energy() - reference.energy()))
                    )
                    assert coefficient_error < 2e-8 and energy_error < 2e-10, (
                        coefficient_error,
                        energy_error,
                    )
                    transfers.append(
                        dict(
                            degree=degree,
                            motion=motion,
                            iterations=iterations,
                            shared=shared,
                            coefficient_error=coefficient_error,
                            energy_error=energy_error,
                        )
                    )
    return dict(products=products, transfers=transfers)


def timing():
    from graded_study import GradedMesh
    from moving_study import profile_family
    from energy_mesh import from_fe

    space = Space(GradedMesh(128, 0.95), 4)
    initial = space.project(profile_family(0.01, 4, spread=0.15), degree=20)
    edges, _ = from_fe(space, initial, 64)
    carrier = Carrier(space, n=64, edges=edges)
    original = MemoryTransport(carrier, initial)
    original.advect(carrier.transport(0.03, "shear").apply_carrier)
    target = Carrier(
        space, n=64, edges=edges + 0.01 * cp.sin(2 * cp.pi * edges) / (2 * cp.pi)
    )
    grid = TensorGrid(target, target.edges)
    cross = lambda w: grid.remap(carrier, w)
    records = []
    for iterations in (0, 2, 4):
        for shared in (False, True):
            samples = {"einsum": [], "reduction": []}
            outputs = {}
            first_reference = None
            ranks = {}
            for backend in samples:
                copy(original).project_memory_galerkin(
                    target, cross, iterations, shared, backend
                )
            for repeat in range(5):
                for backend in (
                    ("einsum", "reduction")
                    if repeat % 2 == 0
                    else ("reduction", "einsum")
                ):
                    state = copy(original)
                    cp.cuda.Stream.null.synchronize()
                    start = time.perf_counter()
                    diagnostic = state.project_memory_galerkin(
                        target, cross, iterations, shared, backend
                    )
                    cp.cuda.Stream.null.synchronize()
                    samples[backend].append(time.perf_counter() - start)
                    outputs[backend] = state
                    ranks[backend] = diagnostic["retained_rank"].tolist()
                    if backend == "einsum" and first_reference is None:
                        first_reference = state
            error = max(
                float(
                    cp.max(
                        cp.abs(
                            getattr(outputs["einsum"], key)
                            - getattr(outputs["reduction"], key)
                        )
                    )
                )
                for key in ("anchor", "auxiliary")
            )
            baseline_noise = max(
                float(
                    cp.max(
                        cp.abs(
                            getattr(outputs["einsum"], key)
                            - getattr(first_reference, key)
                        )
                    )
                )
                for key in ("anchor", "auxiliary")
            )
            difference = copy(outputs["einsum"])
            difference.anchor = outputs["einsum"].anchor - outputs["reduction"].anchor
            difference.auxiliary = (
                outputs["einsum"].auxiliary - outputs["reduction"].auxiliary
            )
            ambient_error = float(cp.sqrt(cp.max(difference.energy()).clip(0)))
            energy_error = float(
                cp.max(
                    cp.abs(outputs["einsum"].energy() - outputs["reduction"].energy())
                )
            )
            record = dict(
                iterations=iterations,
                shared=shared,
                coefficient_error=error,
                reference_repeat_coefficient_noise=baseline_noise,
                ambient_error=ambient_error,
                energy_error=energy_error,
                ranks=ranks,
                seconds=samples,
                median_seconds={k: statistics.median(v) for k, v in samples.items()},
            )
            print(json.dumps(record), flush=True)
            # Near-null Gram directions can amplify coefficient differences;
            # compare the physically represented ambient state and its energy.
            assert ambient_error < 1e-8 and energy_error < 2e-10, record
            assert ranks["einsum"] == ranks["reduction"], record
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    meta = provenance()
    result = validate()
    args.output.write_text(json.dumps(dict(**result, provenance=meta), indent=2) + "\n")
    if not args.checks_only:
        result["timings"] = timing()
    args.output.write_text(json.dumps(dict(**result, provenance=meta), indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "transfers"}), flush=True)


if __name__ == "__main__":
    main()
