"""Cellular transport with graded tensor supports for each oriented mode.

Distributed under the Boost Software License, Version 1.0.
The base carrier retains its state. Each diagonal mode is I+C*(S-I)C, where
C is a transverse-Gauss-row remap into a separately graded coordinate grid.
This is a discrete contraction construction; time order two is not inferred
from the palindromic ordering because projected shears are not reversible.
"""

from types import SimpleNamespace

from gpu import cp
from tensor_remap import TensorTransfer


class TensorCellular:
    def __init__(
        self, carrier, duration, grading=0.95, center=None, oriented_edges=None
    ):
        c = self.carrier = carrier
        coordinates = [
            (c.edges[a, :-1, None] + cp.diff(c.edges[a])[:, None] * c.nodes).ravel()
            for a in (0, 1)
        ]
        self.x = c.shears([(0, duration / 2, 0.5 * cp.sin(2 * cp.pi * coordinates[1]))])
        self.y = c.shears(
            [(1, duration / 2, -0.5 * cp.sin(2 * cp.pi * coordinates[0]))]
        )
        self.diagonal = []
        self.coordinate_maps = []
        location = cp.asarray([0.35, 0.45] if center is None else center)
        for index, (sign, fraction) in enumerate(((1, 0.5), (-1, 1.0))):
            center = (location[0] + sign * location[1]) % 1
            t = cp.linspace(0.0, 1.0, c.n + 1)
            ex = t + grading / (2 * cp.pi) * cp.sin(2 * cp.pi * (t + 0.5 - center))
            if oriented_edges is not None:
                ex = oriented_edges[index]
            edges = cp.stack((ex, c.edges[1]))
            grid = SimpleNamespace(
                n=c.n,
                np=c.np,
                nc=c.nc,
                nodes=c.nodes,
                weights=c.weights,
                module=c.module,
                edges=edges,
            )
            forward = TensorTransfer(
                grid, [(0, 1.0, sign * coordinates[1])], source_edges=c.edges
            )
            adjoint = TensorTransfer(
                c, [(0, 1.0, -sign * coordinates[1])], source_edges=edges
            )
            xi = (ex[:-1, None] + cp.diff(ex)[:, None] * c.nodes).ravel()
            shear = TensorTransfer(
                grid, [(1, duration * fraction, 0.25 * cp.sin(2 * cp.pi * xi))]
            )
            self.diagonal.append((forward, shear, adjoint))
            self.coordinate_maps.append((grid, forward, adjoint))

    def apply_carrier(self, w):
        w = self.y.apply_carrier(self.x.apply_carrier(w))
        for i in (0, 1, 0):
            forward, shear, adjoint = self.diagonal[i]
            old = forward.apply_carrier(w)
            w = w + adjoint.apply_carrier(shear.apply_carrier(old) - old)
        return self.x.apply_carrier(self.y.apply_carrier(w))
