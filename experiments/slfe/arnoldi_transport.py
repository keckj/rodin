"""GPU Arnoldi exponential action for the existing triangular DG generator.

Distributed under the Boost Software License, Version 1.0.
Classical Krylov projection with two orthogonalization passes. The constant
mode is retained explicitly. The projected exponential uses CuPy dense algebra.
The endpoint residual is an indicator, not a certified integrated error bound.
"""

from cupyx.scipy.linalg import expm

from gpu import cp


class ArnoldiTransport:
    def __init__(self, flow, dimension=48):
        if dimension < 2:
            raise ValueError("The Krylov dimension must be at least two")
        self.flow = flow
        self.space = flow.space
        self.size = self.space.mesh.cells * self.space.nb
        self.dimension = min(dimension, self.size - 1)
        self.root_mass = cp.sqrt(self.space.mesh.areas)[:, None, None]
        unit = cp.zeros((self.space.mesh.cells, 1, self.space.nb))
        unit[:, :, 0] = self.root_mass[:, :, 0]
        self.unit = unit.ravel() / cp.linalg.norm(unit)
        self.history = []

    def apply(self, vector):
        coefficients = (
            vector.reshape(self.space.mesh.cells, 1, self.space.nb) / self.root_mass
        )
        return (self.flow.rhs(coefficients) * self.root_mass).ravel()

    def step(self, coefficients, duration):
        if duration < 0:
            raise ValueError("This dissipative DG comparator uses forward time")
        result = cp.empty_like(coefficients)
        for field in range(coefficients.shape[1]):
            value = (coefficients[:, field : field + 1] * self.root_mass).ravel()
            mean = cp.dot(self.unit, value)
            fluctuation = value - mean * self.unit
            norm = cp.linalg.norm(fluctuation)
            if float(norm) < 1e-30:
                result[:, field : field + 1] = coefficients[:, field : field + 1]
                self.history.append(dict(dimension=0, residual_indicator=0.0))
                continue
            basis = cp.empty((self.size, self.dimension + 1), order="F")
            hessenberg = cp.zeros((self.dimension + 1, self.dimension))
            basis[:, 0] = fluctuation / norm
            used = self.dimension
            for j in range(self.dimension):
                vector = self.apply(basis[:, j])
                # The exact DG generator preserves the orthogonal complement
                # of the constant. This removes floating-point drift there.
                vector -= self.unit * cp.dot(self.unit, vector)
                for _ in range(2):
                    projection = basis[:, : j + 1].T @ vector
                    hessenberg[: j + 1, j] += projection
                    vector -= basis[:, : j + 1] @ projection
                beta = cp.linalg.norm(vector)
                hessenberg[j + 1, j] = beta
                if float(beta) <= 1e-14 * max(1.0, self.flow.norm_bound):
                    used = j + 1
                    break
                basis[:, j + 1] = vector / beta
            exponential = expm(duration * hessenberg[:used, :used])[:, 0]
            updated = mean * self.unit + norm * (basis[:, :used] @ exponential)
            result[:, field : field + 1] = (
                updated.reshape(self.space.mesh.cells, 1, self.space.nb)
                / self.root_mass
            )
            self.history.append(
                dict(
                    dimension=used,
                    residual_indicator=float(cp.abs(norm * beta * exponential[-1])),
                )
            )
        return result
