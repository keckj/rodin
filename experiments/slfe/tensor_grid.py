"""Tensor DG geometry without an FE cross-mass matrix.

Distributed under the Boost Software License, Version 1.0.
Two axis projections provide exact tensor-product remapping between grids.
The numerical kernels and Gauss data are borrowed from a compatible carrier.
"""

from gpu import cp
from tensor_remap import TensorTransfer


class TensorGrid:
    def __init__(self, carrier, edges):
        self.n, self.np, self.nc = edges.shape[1] - 1, carrier.np, carrier.nc
        self.nodes, self.weights, self.module = (
            carrier.nodes,
            carrier.weights,
            carrier.module,
        )
        self.edges = cp.ascontiguousarray(edges, dtype=cp.float64)
        if self.edges.shape != (2, self.n + 1) or not bool(
            cp.all(cp.diff(self.edges, axis=1) > 0)
        ):
            raise ValueError("Expected two strictly increasing tensor partitions")
        if not bool(cp.all(cp.abs(self.edges[:, -1] - self.edges[:, 0] - 1) < 1e-13)):
            raise ValueError("Expected periodic partitions of length one")
        h = cp.diff(self.edges, axis=1)
        self.mass = cp.ascontiguousarray(
            (h[1, :, None] * h[0, None, :]).ravel()[:, None]
            * (self.weights[:, None] * self.weights[None, :]).ravel()
        )

    def energy(self, values):
        return cp.sum(self.mass[:, None, :] * values**2, axis=(0, 2))

    def shears(self, stages):
        return TensorTransfer(self, stages)

    def remap(self, source, values):
        if source.np != self.np:
            raise ValueError("This remap retains the polynomial degree")
        if source.n != self.n:
            from tensor_projection import TensorProjection

            return TensorProjection(source, self).apply_carrier(values)
        intermediate = TensorGrid(self, cp.stack((self.edges[0], source.edges[1])))
        zero = cp.zeros(self.n * self.np)
        x = TensorTransfer(intermediate, [(0, 0.0, zero)], source_edges=source.edges)
        y = TensorTransfer(self, [(1, 0.0, zero)], source_edges=intermediate.edges)
        return y.apply_carrier(x.apply_carrier(values))
