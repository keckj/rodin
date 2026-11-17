"""Second rational graph chart, evaluated only through CUDA cross-mass actions.

Distributed under the Boost Software License, Version 1.0.
With A=R*R and Q=I+6A+A^2, the FE injection is
J=[4 R(I+A)Q^-1; (I-A)^2 Q^-1]. Its squared norm is exactly I.
This is the ordinary graph construction applied to T=2R(I+A)^-1.
The change squares the old hidden spectral factor before normalization.
It is a different FE coupling, not a physical L2 reaction extension.
"""

import math

from gpu import cp
from graph_memory import inverse_graph_mass


class SharpenedGraphMemory:
    def __init__(self, coupling, coefficients, degree=48):
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Expected a positive Chebyshev degree")
        self.coupling, self.degree = coupling, degree
        self.value = coefficients.copy()
        temporary = self.solve(coefficients + self.gram(coefficients))
        self.auxiliary = 4 * coupling.project(temporary)
        self.hidden_fe = 2 * (temporary + self.gram(temporary)) - coefficients

    def gram(self, value):
        return self.coupling.adjoint(self.coupling.project(value))

    def solve(self, rhs):
        # Q=(A+3-2sqrt(2)I)(A+3+2sqrt(2)I). The two spectral
        # intervals have condition numbers below 6.83 and 1.18.
        # Degree 12 for the second factor resolves its Chebyshev
        # residual below 1e-16. The first degree is an explicit parameter.
        for shift, degree in (
            (3 - 2 * math.sqrt(2), self.degree),
            (3 + 2 * math.sqrt(2), 12),
        ):
            previous = cp.zeros_like(rhs)
            value = rhs / (shift + 0.5)
            scaled_center = 2 * shift + 1
            t_previous, t_current = 1.0, scaled_center
            for _ in range(1, degree):
                t_next = 2 * scaled_center * t_current - t_previous
                alpha, beta = 4 * t_current / t_next, t_previous / t_next
                action = shift * value + self.gram(value)
                new = value + alpha * (rhs - action) + beta * (value - previous)
                previous, value = value, new
                t_previous, t_current = t_current, t_next
            rhs = value
        return cp.ascontiguousarray(rhs)

    def observe(self):
        r = self.coupling.adjoint(self.auxiliary)
        return self.hidden_fe + 4 * self.solve(r + self.gram(r - 2 * self.hidden_fe))

    def energy(self):
        return self.coupling.energy(self.auxiliary) + self.coupling.space.energy(
            self.hidden_fe
        )

    def canonical_hidden(self):
        temporary = inverse_graph_mass(self.coupling, self.value + self.hidden_fe, 20)
        return 2 * self.coupling.project(temporary) - self.auxiliary

    def advect(self, operation):
        new = operation(self.auxiliary)
        temporary = self.solve(self.coupling.adjoint(new - self.auxiliary))
        self.auxiliary = new
        self.value = self.value + 4 * (temporary + self.gram(temporary))
        return self.value

    def apply_physics(self, operation, hidden_decay=1.0):
        if not 0 <= float(hidden_decay) <= 1:
            raise ValueError("Expected scalar hidden decay in [0,1]")
        new = operation(self.value)
        delta = new - hidden_decay * self.value
        temporary = self.solve(delta + self.gram(delta))
        self.auxiliary = hidden_decay * self.auxiliary + 4 * self.coupling.project(
            temporary
        )
        self.hidden_fe = (
            hidden_decay * self.hidden_fe
            + 2 * (temporary + self.gram(temporary))
            - delta
        )
        self.value = new
        return self.value

    def transfer(self, coupling, cross_projection):
        if coupling.space is not self.coupling.space:
            raise ValueError("Graph transfer requires the same FE space")
        before = self.energy()
        hidden = self.canonical_hidden()
        moved = cross_projection(hidden)
        loss = self.coupling.energy(hidden) - coupling.energy(moved)
        self.coupling = coupling
        temporary = self.solve(
            2 * coupling.adjoint(moved) + self.value + self.gram(self.value)
        )
        self.auxiliary = 4 * coupling.project(temporary) - moved
        self.hidden_fe = 2 * (temporary + self.gram(temporary)) - self.value
        return dict(energy_loss=before - self.energy(), hidden_transfer_loss=loss)
