"""Orthogonal DG splitting with rank-two correction of the constant mode.

Distributed under the Boost Software License, Version 1.0.
For a local orthogonal U and normalized constant c, the minimal plane rotation
Q maps Uc back to c. Q*U is orthogonal, fixes c, is symmetric in time, and has
infinitesimal generator (I-cc*) A (I-cc*) when U=exp(t*A). The sum of these
projected generators equals A whenever the complete DG operator annihilates c.
Local groups contain less than half the constant's mass, preventing antipodes.
"""

from gpu import cp, HERE, launch
from eulerian_dg import EulerianDG


class RotationDG:
    def __init__(self, space):
        self.space = space
        self.dg = EulerianDG(space, flux="central")
        self.root = cp.sqrt(space.mesh.areas)[:, None, None]
        self.constant = cp.zeros((space.mesh.cells, 1, space.nb))
        self.constant[:, :, 0] = self.root[:, :, 0]
        self.kernel = cp.RawKernel(
            f"#define NB {space.nb}\n" + (HERE / "rotation_dg.cu").read_text(),
            "local_rotation",
        )
        # Color the dual edges into disjoint matchings. The line graph has
        # maximum degree four, so five colors suffice for a greedy coloring.
        left, edge = cp.nonzero(cp.arange(space.mesh.cells)[:, None] < self.dg.neighbor)
        right = self.dg.neighbor[left, edge]
        ids = cp.arange(len(left), dtype=cp.int32)
        incident = cp.empty((space.mesh.cells, 3), dtype=cp.int32)
        incident[left, edge] = ids
        reverse = cp.argmax(self.dg.neighbor[right] == left[:, None], axis=1)
        incident[right, reverse] = ids
        neighbors = cp.concatenate((incident[left], incident[right]), axis=1)
        priority = (ids.astype(cp.uint64) * 2654435761) % 2**32
        colors = cp.full(len(left), -1, dtype=cp.int32)
        rounds = 0
        while bool(cp.any(colors < 0)):
            eligible = (colors < 0) & cp.all(
                (colors[neighbors] >= 0) | (priority[:, None] <= priority[neighbors]),
                axis=1,
            )
            available = cp.stack(
                [cp.all(colors[neighbors] != i, axis=1) for i in range(5)], axis=1
            )
            chosen = cp.argmax(available, axis=1).astype(cp.int32)
            assert bool(cp.all(cp.any(available, axis=1) | (colors >= 0)))
            colors = cp.where(eligible, chosen, colors)
            rounds += 1
            if rounds > len(left):
                raise RuntimeError("Edge coloring failed to progress")
        self.coloring_rounds = rounds
        self.groups = []
        volume_indices = cp.arange(space.mesh.cells, dtype=cp.int32)[:, None]
        self.add_groups(volume_indices, self.dg.diagonal)
        nb = space.nb
        for color in range(5):
            active = cp.flatnonzero(colors == color)
            if not len(active):
                continue
            indices = cp.stack((left[active], right[active]), axis=1).astype(cp.int32)
            assert len(cp.unique(indices)) == indices.size
            cross = (
                self.dg.offdiagonal[left[active], edge[active]]
                * self.root[left[active]]
                / self.root[right[active]]
            )
            blocks = cp.zeros((len(active), 2 * nb, 2 * nb))
            blocks[:, :nb, nb:] = cross
            blocks[:, nb:, :nb] = -cross.transpose(0, 2, 1)
            self.add_groups(indices, blocks)

    def add_groups(self, indices, blocks):
        weights = cp.sum(self.space.mesh.areas[indices], axis=1)
        largest = float(cp.max(weights))
        # A group with constant mass mu < 1/2 has <c,Uc> >= 1-2*mu > 0
        # for every duration. Reserve 1e-12 against binning roundoff.
        width = 0.5 - largest - 1e-12
        if width <= 0:
            raise ValueError("A local block covers half the domain or more")
        bins = cp.floor((cp.cumsum(weights) - weights) / width).astype(cp.int32)
        for bin_id in range(int(cp.max(bins)) + 1):
            selected = cp.flatnonzero(bins == bin_id)
            if not len(selected):
                continue
            mass = float(cp.sum(weights[selected]))
            assert mass < 0.5
            values, vectors = cp.linalg.eigh(1j * blocks[selected])
            self.groups.append(
                (cp.ascontiguousarray(indices[selected]), values, vectors, mass)
            )

    def local_apply(self, indices, matrix, value):
        result = value.copy()
        launch(
            self.kernel,
            indices.size * value.shape[1] * self.space.nb,
            (
                indices,
                matrix,
                value,
                result,
                len(indices),
                indices.shape[1],
                value.shape[1],
            ),
        )
        return result

    def prepare(self, duration, order=4):
        return PreparedRotation(self, duration, order)


class PreparedRotation:
    """Fixed-duration propagator; its step argument is only a graph interface."""

    def __init__(self, flow, duration, order):
        if order not in (2, 4):
            raise ValueError("Implemented orders are 2 and 4")
        self.flow = flow
        self.duration = duration
        self.stages = []
        ng = len(flow.groups)
        palindrome = (
            [(i, 0.5) for i in range(ng - 1)]
            + [(ng - 1, 1.0)]
            + [(i, 0.5) for i in reversed(range(ng - 1))]
        )
        a = 1 / (2 - 2 ** (1 / 3))
        weights = (1.0,) if order == 2 else (a, 1 - 2 * a, a)
        cache = {}
        self.minimum_overlap = 1.0
        for weight in weights:
            for index, fraction in palindrome:
                dt = duration * weight * fraction
                key = (index, dt)
                if key not in cache:
                    indices, values, vectors, mass = flow.groups[index]
                    matrix = cp.ascontiguousarray(
                        (
                            (vectors * cp.exp(-1j * dt * values)[:, None, :])
                            @ vectors.conj().transpose(0, 2, 1)
                        ).real
                    )
                    b = flow.local_apply(indices, matrix, flow.constant)
                    delta = b - flow.constant
                    overlap = float(cp.sum(flow.constant * b))
                    assert overlap > 0, (overlap, mass)
                    self.minimum_overlap = min(self.minimum_overlap, overlap)
                    cache[key] = (indices, matrix, delta, overlap)
                self.stages.append(cache[key])

    def step(self, value, unused_duration=None):
        flow = self.flow
        value = value * flow.root
        for indices, matrix, delta, overlap in self.stages:
            value = flow.local_apply(indices, matrix, value)
            mean = cp.sum(flow.root[:, :, 0] * value[:, :, 0], axis=0)
            difference = cp.sum(delta * value, axis=(0, 2))
            alpha = (2 * (overlap - 1) * mean + 2 * overlap * difference) / (
                1 + overlap
            )
            beta = -(2 * mean + difference) / (1 + overlap)
            value = (
                value
                + flow.constant * alpha[None, :, None]
                + delta * beta[None, :, None]
            )
        return value / flow.root
