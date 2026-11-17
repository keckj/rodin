"""GPU equivalence controls for translation-invariant tensor block storage.

Distributed under the Boost Software License, Version 1.0.
The independent full representation is selected through explicit edges;
it uses the nonuniform geometry path even when those edges are uniform.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from carrier import Carrier
from tensor_remap import TensorTransfer
from campaign import provenance


def storage(operator):
    return sum(
        starts.nbytes + offsets.nbytes + blocks.nbytes
        for _, _, starts, offsets, blocks in operator.operators
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = cp.random.RandomState(914)
    records = []
    for degree in (1, 2, 4):
        space = Space(Mesh(4, jitter=0.13, seed=194), degree)
        for n in (3, 7, 16):
            compact = Carrier(space, n=n, builder="tiled")
            full = Carrier(space, n=n, edges=compact.edges, builder="tiled")
            value = rng.normal(size=(n * n, 3, compact.nc))
            for axis in (0, 1):
                coordinate = (
                    cp.arange(n)[:, None] + compact.nodes[None, :]
                ).ravel() / n
                speed = 0.37 + 0.91 * cp.sin(2 * cp.pi * coordinate)
                for duration in (0.0, 1e-7, -1e-7, 0.07, -0.31, 1.0, -3.0):
                    a = TensorTransfer(compact, [(axis, duration, speed)])
                    b = TensorTransfer(full, [(axis, duration, speed)])
                    actual = a.apply_carrier(value)
                    expected = b.apply_carrier(value)
                    error = float(cp.max(cp.abs(actual - expected)))
                    constant_error = float(
                        cp.max(cp.abs(a.apply_carrier(cp.ones_like(value)) - 1))
                    )
                    mass_error = float(
                        cp.max(
                            cp.abs(cp.sum((actual - value) * compact.mass, axis=(0, 2)))
                        )
                    )
                    ratio = float(
                        cp.max(compact.energy(actual) / compact.energy(value))
                    )
                    assert max(error, constant_error, mass_error) < 2e-11, (
                        degree,
                        n,
                        axis,
                        duration,
                        error,
                    )
                    assert ratio <= 1 + 1e-11, ratio
                    records.append(
                        dict(
                            degree=degree,
                            n=n,
                            axis=axis,
                            duration=duration,
                            max_difference=error,
                            constant_error=constant_error,
                            mass_error=mass_error,
                            energy_ratio=ratio,
                            compact_bytes=storage(a),
                            full_bytes=storage(b),
                        )
                    )
            # Source-edge remaps must retain the full representation even for
            # a uniform destination: the source is a different partition.
            edges = compact.edges.copy()
            edges[0] += 0.03 * cp.sin(2 * cp.pi * edges[0])
            a = TensorTransfer(compact, [(0, 0.07, speed)], source_edges=edges)
            b = TensorTransfer(full, [(0, 0.07, speed)], source_edges=edges)
            error = float(
                cp.max(cp.abs(a.apply_carrier(value) - b.apply_carrier(value)))
            )
            assert error < 1e-12 and not a.operators[0][1]
            records.append(dict(degree=degree, n=n, changed_source_difference=error))
            for obj in (a, b):
                del obj
            cp.get_default_memory_pool().free_all_blocks()
    args.output.write_text(
        json.dumps(dict(provenance=provenance(), controls=records), indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                controls=len(records),
                max_difference=max(x.get("max_difference", 0) for x in records),
            )
        )
    )


if __name__ == "__main__":
    main()
