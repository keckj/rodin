"""Nonuniform tensor DG shears with exact overlap quadrature on CUDA.

Distributed under the Boost Software License, Version 1.0.
The transverse direction is represented at Gauss nodes. Conservation and
contraction hold in the exact tensor polynomial mass norm. Dependence on the
transverse coordinate is a collocation approximation to the continuous shear.
"""

from gpu import cp, launch


class TensorTransfer:
    def __init__(self, carrier, stages, source_edges=None):
        self.carrier = carrier
        self.operators = []
        self.kernel = carrier.module.get_function("tensor_apply")
        c = carrier
        if source_edges is not None and len(stages) != 1:
            raise ValueError("A change of partition requires exactly one shear")
        for axis, dt, velocity in stages:
            if axis not in (0, 1) or velocity.shape != (c.n * c.np,):
                raise ValueError("Expected axis 0/1 and one velocity per Gauss row")
            velocity = cp.ascontiguousarray(velocity, dtype=cp.float64)
            edges = cp.ascontiguousarray(c.edges[axis])
            source = (
                edges
                if source_edges is None
                else cp.ascontiguousarray(source_edges[axis])
            )
            if source_edges is not None and not bool(
                cp.all(source_edges[1 - axis] == c.edges[1 - axis])
            ):
                raise ValueError("Transverse partitions must coincide for a row remap")
            duration = cp.asarray(dt, dtype=cp.float64)
            rows_count = c.n**2 * c.np
            starts = cp.empty(rows_count, dtype=cp.int32)
            counts = cp.empty(rows_count, dtype=cp.int64)
            launch(
                c.module.get_function("tensor_counts"),
                rows_count,
                (edges, source, c.n, velocity, duration, starts, counts),
            )
            offsets = cp.concatenate((cp.zeros(1, dtype=cp.int64), cp.cumsum(counts)))
            pairs = int(offsets[-1])
            rows = cp.searchsorted(offsets[1:], cp.arange(pairs), side="right").astype(
                cp.int32
            )
            blocks = cp.empty((pairs, c.np, c.np), dtype=cp.float64)
            launch(
                c.module.get_function("tensor_blocks"),
                blocks.size,
                (
                    edges,
                    source,
                    c.n,
                    velocity,
                    duration,
                    starts,
                    offsets,
                    rows,
                    cp.int64(pairs),
                    c.nodes,
                    c.weights,
                    blocks,
                ),
            )
            self.operators.append((axis, starts, offsets, blocks))

    def apply_carrier(self, coefficients):
        c = self.carrier
        for axis, starts, offsets, blocks in self.operators:
            output = cp.empty_like(coefficients)
            launch(
                self.kernel,
                output.size,
                (
                    coefficients,
                    output,
                    c.n,
                    output.shape[1],
                    axis,
                    starts,
                    offsets,
                    blocks,
                ),
            )
            coefficients = output
        return coefficients

    def apply(self, coefficients, incremental=True):
        c = self.carrier
        old = c.project(coefficients)
        new = self.apply_carrier(old)
        return coefficients + c.adjoint(new - old) if incremental else c.adjoint(new)
