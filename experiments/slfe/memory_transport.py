"""A persistent auxiliary state with contractive FE physics updates.

Distributed under the Boost Software License, Version 1.0.
No square root or inverse of a projection defect is evaluated at runtime.
"""


class MemoryTransport:
    def __init__(self, coupling, coefficients):
        self.coupling = coupling
        self.value = coefficients.copy()
        self.auxiliary = coupling.project(coefficients)

    def advect(self, operation):
        new = operation(self.auxiliary)
        self.value = self.value + self.coupling.adjoint(new - self.auxiliary)
        self.auxiliary = new
        return self.value

    def apply_physics(self, operation):
        new = operation(self.value)
        self.auxiliary = self.auxiliary + self.coupling.project(new - self.value)
        self.value = new
        return self.value
