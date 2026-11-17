"""Conservative projected shears for periodic Fourier streamfunctions on CUDA.

Distributed under the Boost Software License, Version 1.0.
An integer area-preserving chart sends each primitive wavevector to the
first coordinate. Its Fourier harmonics generate one exact transverse
shear. Coordinate projections C enter as I+C*(S-I)C, which is contractive
when C and S are contractions. This prototype requires a uniform tensor
support. FE streamfunctions can supply the retained Fourier coefficients.
The finite set of retained modes is an explicit velocity approximation.
"""

import math

from gpu import cp, quadrature
from tensor_remap import TensorTransfer


def integer_chart(k, l):
    """Return integer shears and quarter-turns for a unimodular chart.

    Integer Euclid operations are symbolic geometry setup. All field
    transforms and numerical coefficient calculations remain on CUDA.
    A positive quarter-turn here is (x,y) -> (y,-x).
    """
    if math.gcd(k, l) != 1:
        raise ValueError("Expected a primitive nonzero wavevector")
    r0, r1, s0, s1, t0, t1 = k, l, 1, 0, 0, 1
    while r1:
        quotient = r0 // r1
        r0, r1 = r1, r0 - quotient * r1
        s0, s1 = s1, s0 - quotient * s1
        t0, t1 = t1, t0 - quotient * t1
    if r0 < 0:
        s0, t0 = -s0, -t0
    a, b, c, d = k, l, -t0, s0
    matrix = (a, b, c, d)
    left = []
    while c:
        quotient = a // c
        if quotient:
            a, b = a - quotient * c, b - quotient * d
            left.append(("shear", -quotient))
        a, b, c, d = c, d, -a, -b
        left.append(("rotation", 1))
    if a < 0:
        a, b, c, d = -a, -b, -c, -d
        left.append(("rotation", 2))
    if b:
        left.append(("shear", -b))
        a, b = a - b * c, b - b * d
    assert (a, b, c, d) == (1, 0, 0, 1)
    operations = []
    for kind, value in reversed(left):
        value = -value
        if operations and operations[-1][0] == kind:
            value += operations.pop()[1]
        if kind == "rotation":
            value %= 4
        if value:
            operations.append((kind, value))
    # Independent integer reconstruction catches order/sign errors in setup.
    a, b, c, d = 1, 0, 0, 1
    for kind, value in operations:
        if kind == "shear":
            a, b = a + value * c, b + value * d
        else:
            for _ in range(value):
                a, b, c, d = c, d, -a, -b
    assert (a, b, c, d) == matrix
    return operations, matrix


def streamfunction_modes(space, value, wavevectors, subdivisions=1):
    """Compute selected real Fourier moments of an FE streamfunction on CUDA."""
    if value.shape != (space.mesh.cells, 1, space.nb):
        raise ValueError("Expected one scalar FE streamfunction")
    q = quadrature(20, subdivisions)
    sampled = cp.einsum("cb,qb->cq", value[:, 0], space.basis(q[:, :2]))
    sampled *= 2 * space.mesh.areas[:, None] * q[None, :, 2]
    points = space.mesh.points(q)
    result = []
    for k, l in wavevectors:
        angle = 2 * cp.pi * (k * points[..., 0] + l * points[..., 1])
        result.append(
            cp.stack(
                (
                    2 * cp.sum(sampled * cp.cos(angle)),
                    2 * cp.sum(sampled * cp.sin(angle)),
                )
            )
        )
    return cp.ascontiguousarray(cp.stack(result))


class TensorHamiltonian:
    def __init__(
        self, carrier, wavevectors, coefficients, duration, coordinate_maps=None
    ):
        if carrier.nonuniform:
            raise ValueError(
                "Integer coordinate permutations require uniform tensor supports"
            )
        if (
            coefficients.shape != (len(wavevectors), 2)
            or coefficients.dtype != cp.float64
        ):
            raise ValueError(
                "Expected CUDA cosine/sine coefficients for each wavevector"
            )
        self.carrier = carrier
        c = carrier
        coordinate = ((cp.arange(c.n)[:, None] + c.nodes) / c.n).ravel()
        groups = {}
        for (k, l), coefficient in zip(wavevectors, coefficients):
            harmonic = math.gcd(k, l)
            if not harmonic:
                raise ValueError("The constant streamfunction generates no velocity")
            k, l = k // harmonic, l // harmonic
            cosine, sine = coefficient[0], coefficient[1]
            if k < 0 or (k == 0 and l < 0):
                k, l, sine = -k, -l, -sine
            angle = 2 * cp.pi * harmonic * coordinate
            speed = (
                2 * cp.pi * harmonic * (cosine * cp.sin(angle) - sine * cp.cos(angle))
            )
            groups[k, l] = groups.get((k, l), 0) + speed
        self.charts, self.matrices = [], []
        coordinate_maps = {} if coordinate_maps is None else coordinate_maps
        for index, ((k, l), speed) in enumerate(groups.items()):
            key = (id(c), k, l)
            if key not in coordinate_maps:
                setup, matrix = integer_chart(k, l)
                operators = []
                for kind, value in setup:
                    if kind == "rotation":
                        operators.append((kind, value, (-value) % 4))
                    else:
                        operators.append(
                            (
                                kind,
                                TensorTransfer(c, [(0, 1.0, value * coordinate)]),
                                TensorTransfer(c, [(0, 1.0, -value * coordinate)]),
                            )
                        )
                coordinate_maps[key] = (operators, matrix)
            operators, matrix = coordinate_maps[key]
            self.matrices.append(matrix)
            fraction = 1.0 if index == len(groups) - 1 else 0.5
            shear = TensorTransfer(c, [(1, duration * fraction, speed)])
            self.charts.append((operators, shear))

    def transform(self, value, operators, adjoint=False):
        c = self.carrier
        for kind, forward, backward in reversed(operators) if adjoint else operators:
            operation = backward if adjoint else forward
            if kind == "shear":
                value = operation.apply_carrier(value)
                continue
            a = value.reshape(c.n, c.n, value.shape[1], c.np, c.np)
            if operation == 1:
                a = a.transpose(1, 0, 2, 4, 3)[::-1, :, :, ::-1, :]
            elif operation == 2:
                a = a[::-1, ::-1, :, ::-1, ::-1]
            elif operation == 3:
                a = a.transpose(1, 0, 2, 4, 3)[:, ::-1, :, :, ::-1]
            value = cp.ascontiguousarray(a.reshape(value.shape))
        return value

    def apply_carrier(self, value):
        sequence = list(range(len(self.charts))) + list(
            range(len(self.charts) - 2, -1, -1)
        )
        for index in sequence:
            operators, shear = self.charts[index]
            if not operators:
                value = shear.apply_carrier(value)
            else:
                mapped = self.transform(value, operators)
                increment = shear.apply_carrier(mapped) - mapped
                value = value + self.transform(increment, operators, adjoint=True)
        return value


class ComposedHamiltonian:
    """Classical higher-order composition candidates with contractive stages.

    The continuous shear composition is symmetric. Its discrete projections
    are not reversible, so the formal continuous order is a hypothesis to
    test after spatial discretization. Each signed-duration stage remains
    mass preserving and contractive; this stability does not imply the order.
    """

    def __init__(self, carrier, wavevectors, coefficients, duration, scheme):
        if scheme == "triple4":
            positive = 1 / (2 - 2 ** (1 / 3))
            weights = [positive, 1 - 2 * positive, positive]
        elif scheme == "suzuki4":
            positive = 1 / (4 - 4 ** (1 / 3))
            weights = [positive, positive, 1 - 4 * positive, positive, positive]
        elif scheme == "yoshida6":
            # Yoshida (1990), Table 1, solution A; equation (5.1).
            w1, w2, w3 = -1.177679984178870, 0.235573213359357, 0.784513610477560
            w0 = 1 - 2 * (w1 + w2 + w3)
            weights = [w3, w2, w1, w0, w1, w2, w3]
        else:
            raise ValueError("Expected triple4, suzuki4 or yoshida6 composition")
        self.weights = tuple(weights)
        # Coordinate projections do not depend on the signed substep. Share
        # their assembled blocks across stages without altering arithmetic.
        coordinate_maps = {}
        flows = {
            weight: TensorHamiltonian(
                carrier, wavevectors, coefficients, duration * weight, coordinate_maps
            )
            for weight in set(weights)
        }
        self.sequence = [flows[weight] for weight in weights]

    def apply_carrier(self, value):
        for flow in self.sequence:
            value = flow.apply_carrier(value)
        return value
