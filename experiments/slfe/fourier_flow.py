"""Volume-preserving oriented shears for a nonseparable cellular flow.

Distributed under the Boost Software License, Version 1.0.
The carrier coordinates are integer unimodular maps of the periodic torus.
This is a controlled four-mode experiment, not an arbitrary-velocity solver.
"""

from types import SimpleNamespace
import time

from gpu import cp, Space
from carrier import Carrier, IncrementalTransfer


class CellularTransport:
    """Compose incremental transports for four Hamiltonian Fourier modes.

    The base streamfunction is sin(pi*x)^2*sin(pi*y)^2/pi. Its velocity is
    exactly the one used in flows.cellular_velocity, before the time factor.
    The splitting order is symmetric in mode durations, but the projected
    increments are not reversible maps. Second-order time convergence is not
    inferred from the symmetric ordering.
    """

    def __init__(self, space, factor=2, degree=None, grouped=True):
        self.space = space
        self.grouped = grouped
        self.carriers = []
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        for matrix in (((1, 0), (0, 1)), ((1, 1), (0, 1)), ((1, -1), (0, 1))):
            transform = cp.asarray(matrix, dtype=cp.float64)
            triangles = cp.ascontiguousarray(space.mesh.triangles @ transform.T)
            mesh = SimpleNamespace(
                triangles=triangles,
                areas=space.mesh.areas,
                cells=space.mesh.cells,
                n=space.mesh.n,
            )
            self.carriers.append(
                Carrier(
                    Space(mesh, space.degree), n=space.mesh.n * factor, degree=degree
                )
            )
        self.velocities = []
        for carrier in self.carriers:
            coordinate = (
                cp.arange(carrier.n)[:, None] + carrier.nodes[None, :]
            ).ravel() / carrier.n
            self.velocities.append(cp.sin(2 * cp.pi * coordinate))
        self.vx = 0.5 * self.velocities[0]
        self.vy = -0.5 * self.velocities[0]
        self.vplus = 0.25 * self.velocities[1]
        self.vminus = 0.25 * self.velocities[2]
        cp.cuda.Stream.null.synchronize()
        self.build_seconds = time.perf_counter() - start
        self.storage_bytes = sum(
            c.blocks.nbytes + c.source.nbytes + c.target.nbytes for c in self.carriers
        )

    def step(self, t0, t1, reversible=True):
        # The common time multiplier can be integrated exactly before splitting.
        # Reversing flow has the identity as its exact full-period map.
        duration = (
            (cp.sin(cp.asarray(cp.pi * t1)) - cp.sin(cp.asarray(cp.pi * t0))) / cp.pi
            if reversible
            else t1 - t0
        )
        c0, plus, minus = self.carriers
        half = duration / 2
        cartesian = c0.shears([(0, half, self.vx), (1, half, self.vy)])
        diagonal_plus = plus.shears([(1, half, self.vplus)])
        diagonal_minus = minus.shears([(1, duration, self.vminus)])
        middle = [diagonal_plus, diagonal_minus, diagonal_plus]
        if self.grouped:
            stages = (
                [cartesian]
                + middle
                + [IncrementalTransfer(c0, cartesian.operators[::-1])]
            )
        else:
            x = IncrementalTransfer(c0, cartesian.operators[:1])
            y = IncrementalTransfer(c0, cartesian.operators[1:])
            stages = [x, y] + middle + [y, x]
        return CellularStep(stages)


class CellularStep:
    def __init__(self, stages):
        self.stages = stages

    def apply(self, coefficients):
        for stage in self.stages:
            coefficients = stage.apply(coefficients)
        return coefficients


class NestedCellularTransport:
    """Represent oriented mode changes by structured coordinate remaps.

    An oriented mode is I + C* (S_mode-I) C on one Cartesian carrier.
    C is a static coordinate shear; its mass adjoint is the opposite shear.
    The outer FE increment is then returned once for the whole split step.
    Both levels use the same contraction argument. Coordinate approximation
    and physical accuracy are independent measurements.
    """

    def __init__(self, space, factor=2, degree=None, grouped=True):
        self.space = space
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        self.carrier = Carrier(space, n=space.mesh.n * factor, degree=degree)
        c = self.carrier
        coordinate = (cp.arange(c.n)[:, None] + c.nodes[None, :]).ravel() / c.n
        wave = cp.sin(2 * cp.pi * coordinate)
        self.vx = 0.5 * wave
        self.vy = -0.5 * wave
        self.vdiagonal = 0.25 * wave
        self.plus = c.shears([(0, 1.0, coordinate)])
        self.minus = c.shears([(0, 1.0, -coordinate)])
        cp.cuda.Stream.null.synchronize()
        self.build_seconds = time.perf_counter() - start
        self.storage_bytes = c.blocks.nbytes + c.source.nbytes + c.target.nbytes

    def step(self, t0, t1, reversible=True):
        duration = (
            (cp.sin(cp.asarray(cp.pi * t1)) - cp.sin(cp.asarray(cp.pi * t0))) / cp.pi
            if reversible
            else t1 - t0
        )
        c = self.carrier
        half = duration / 2
        xy = c.shears([(0, half, self.vx), (1, half, self.vy)])
        diagonal_half = c.shears([(1, half, self.vdiagonal)])
        diagonal_full = c.shears([(1, duration, self.vdiagonal)])
        plus = CoordinateIncrement(self.plus, diagonal_half, self.minus)
        minus = CoordinateIncrement(self.minus, diagonal_full, self.plus)
        return NestedCellularStep(
            c, [xy, plus, minus, plus, IncrementalTransfer(c, xy.operators[::-1])]
        )


class CoordinateIncrement:
    def __init__(self, forward, transport, adjoint):
        self.forward = forward
        self.transport = transport
        self.adjoint = adjoint

    def apply_carrier(self, coefficients):
        old = self.forward.apply_carrier(coefficients)
        difference = self.transport.apply_carrier(old) - old
        return coefficients + self.adjoint.apply_carrier(difference)


class NestedCellularStep:
    def __init__(self, carrier, stages):
        self.carrier = carrier
        self.stages = stages

    def apply(self, coefficients):
        old = self.carrier.project(coefficients)
        new = self.apply_carrier(old)
        return coefficients + self.carrier.adjoint(new - old)

    def apply_carrier(self, coefficients):
        new = coefficients
        for stage in self.stages:
            new = stage.apply_carrier(new)
        return new


def validate_decomposition():
    from flows import cellular_velocity

    points = cp.random.RandomState(912).uniform(0, 1, (4096, 2))
    x, y = points.T
    along_x = 0.5 * cp.sin(2 * cp.pi * y)
    along_y = -0.5 * cp.sin(2 * cp.pi * x)
    diagonal_plus = 0.25 * cp.sin(2 * cp.pi * (x + y))
    diagonal_minus = 0.25 * cp.sin(2 * cp.pi * (x - y))
    reconstructed = cp.stack(
        (
            along_x - diagonal_plus + diagonal_minus,
            along_y + diagonal_plus + diagonal_minus,
        ),
        axis=-1,
    )
    error = float(cp.max(cp.abs(reconstructed - cellular_velocity(points, 0))))
    assert error < 2e-15, error
    return error
