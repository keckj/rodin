"""Exact tensor-product DG projection between partitions of different sizes.

Distributed under the Boost Software License, Version 1.0.
GPU overlap quadrature constructs sparse one-dimensional matrices. The two
partitions must share their interval endpoints; periodic wrapping is handled
by the shear kernels, not by this static projection.
"""

from cupyx.scipy.sparse import coo_matrix
from gpu import cp
from flux import gauss_interval


class TensorProjection:
    def __init__(self, source, target, maps=None, quadrature_order=None):
        """Project through optional separable conservative characteristic maps.

        Each maps entry is (forward, inverse) for a monotone interval map
        fixing the partition endpoints. Integration is over source intervals
        split at inverse images of target edges. With nonlinear maps this is
        quadrature, and its accuracy requires an independent refinement test.
        Without maps the original exact polynomial projection is retained.
        """
        if source.np != target.np:
            raise ValueError("This projection retains the polynomial degree")
        if not bool(
            cp.all(cp.abs(source.edges[:, [0, -1]] - target.edges[:, [0, -1]]) < 1e-13)
        ):
            raise ValueError("The partitions must share their interval endpoints")
        self.source, self.target = source, target
        self.matrices = []
        p = source.np
        nodes, weights = source.nodes, source.weights
        qnodes, qweights = gauss_interval(quadrature_order or p)
        for axis in (0, 1):
            se, te = source.edges[axis], target.edges[axis]
            preimage = te if maps is None else maps[axis][1](te)
            if not bool(cp.all(cp.diff(preimage) > 0)) or not bool(
                cp.all(cp.abs(preimage[[0, -1]] - se[[0, -1]]) < 1e-13)
            ):
                raise ValueError("Characteristic maps must increase and fix endpoints")
            lower = cp.maximum(preimage[:-1, None], se[None, :-1])
            upper = cp.minimum(preimage[1:, None], se[None, 1:])
            target_cells, source_cells = cp.nonzero(upper > lower)
            lo, length = (
                lower[target_cells, source_cells],
                (upper - lower)[target_cells, source_cells],
            )
            ht, hs = cp.diff(te)[target_cells], cp.diff(se)[source_cells]
            points = lo[:, None] + length[:, None] * qnodes
            mapped = points if maps is None else maps[axis][0](points)
            tx = (mapped - te[target_cells, None]) / ht[:, None]
            sx = (points - se[source_cells, None]) / hs[:, None]
            bt, bs = cp.ones((*points.shape, p)), cp.ones((*points.shape, p))
            for i in range(p):
                for j in range(p):
                    if i != j:
                        bt[..., i] *= (tx - nodes[j]) / (nodes[i] - nodes[j])
                        bs[..., i] *= (sx - nodes[j]) / (nodes[i] - nodes[j])
            blocks = (
                cp.einsum("q,pqi,pqj,p->pij", qweights, bt, bs, length / ht)
                / weights[None, :, None]
            )
            rows = cp.broadcast_to(
                target_cells[:, None, None] * p + cp.arange(p)[None, :, None],
                blocks.shape,
            )
            columns = cp.broadcast_to(
                source_cells[:, None, None] * p + cp.arange(p)[None, None, :],
                blocks.shape,
            )
            self.matrices.append(
                coo_matrix(
                    (blocks.ravel(), (rows.ravel(), columns.ravel())),
                    shape=(target.n * p, source.n * p),
                ).tocsr()
            )

    def apply_carrier(self, values):
        s, t = self.source, self.target
        p, fields = s.np, values.shape[1]
        values = values.reshape(s.n, s.n, fields, p, p).transpose(0, 3, 1, 4, 2)
        values = values.reshape(s.n * p, s.n * p, fields)
        along_x = self.matrices[0] @ cp.ascontiguousarray(
            values.transpose(1, 0, 2).reshape(s.n * p, -1)
        )
        along_x = cp.ascontiguousarray(
            along_x.reshape(t.n * p, s.n * p, fields).transpose(1, 0, 2)
        )
        result = self.matrices[1] @ along_x.reshape(s.n * p, -1)
        return cp.ascontiguousarray(
            result.reshape(t.n, p, t.n, p, fields)
            .transpose(0, 2, 4, 1, 3)
            .reshape(t.n * t.n, fields, p * p)
        )
