"""Synchronized GPU stage timings for the augmented memory projection.

Distributed under the Boost Software License, Version 1.0.
This instrumentation diagnoses individual calls; it is not a solve benchmark.
"""

import argparse
from copy import copy
import json
from pathlib import Path
import time

from gpu import cp, Space
from graded_study import GradedMesh
from moving_study import profile_family
from energy_mesh import from_fe
from carrier import Carrier
from tensor_grid import TensorGrid
from memory_transport import MemoryTransport
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
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
    einsum, eigh = cp.einsum, cp.linalg.eigh
    records = []
    for iterations in (0, 1, 2, 4):
        for shared in (False, True):
            for _ in range(2):
                copy(original).project_memory_galerkin(
                    target, cross, iterations, shared
                )
            timings = []

            def measured(name, function, *positional, **keywords):
                cp.cuda.Stream.null.synchronize()
                start = time.perf_counter()
                result = function(*positional, **keywords)
                cp.cuda.Stream.null.synchronize()
                timings.append(
                    dict(operation=name, seconds=time.perf_counter() - start)
                )
                return result

            cp.einsum = lambda expression, *values, **options: measured(
                expression, einsum, expression, *values, **options
            )
            cp.linalg.eigh = lambda matrix: measured("eigh", eigh, matrix)
            try:
                cp.cuda.Stream.null.synchronize()
                start = time.perf_counter()
                copy(original).project_memory_galerkin(
                    target, cross, iterations, shared
                )
                cp.cuda.Stream.null.synchronize()
                total = time.perf_counter() - start
            finally:
                cp.einsum, cp.linalg.eigh = einsum, eigh
            result = dict(
                iterations=iterations, shared=shared, seconds=total, timings=timings
            )
            records.append(result)
            print(json.dumps(result), flush=True)
    args.output.write_text(
        json.dumps(dict(results=records, provenance=meta), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
