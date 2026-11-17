"""Linear observable-preserving transport through an orthogonal graph chart.

Distributed under the Boost Software License, Version 1.0.
This is a different coupling from the physical common-L2 realization.
For M=I+R*R, the graph reflection U=2 [R;I] M^-1 [R*,I]-I is orthogonal.
The FE embedding J=U[0;I] has first block 2 R M^-1, not R.
Every inverse has spectrum in [1,2] when R is contractive. A fixed
Chebyshev polynomial evaluates it on CUDA; its finite-degree defect is
measured independently. No inverse of I-R*R is formed.
"""

from gpu import cp


def inverse_graph_mass(coupling, rhs, degree=20):
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("A positive inverse polynomial degree is required")
    previous = cp.zeros_like(rhs)
    value = rhs / 1.5
    t_previous, t_current = 1.0, 3.0
    for _ in range(1, degree):
        t_next = 6 * t_current - t_previous
        alpha, beta = 4 * t_current / t_next, t_previous / t_next
        action = value + coupling.adjoint(coupling.project(value))
        new = value + alpha * (rhs - action) + beta * (value - previous)
        previous, value = value, new
        t_previous, t_current = t_current, t_next
    return cp.ascontiguousarray(value)


class GraphMemory:
    def __init__(self, coupling, coefficients, degree=20):
        self.coupling, self.degree = coupling, degree
        self.value = coefficients.copy()
        temporary = self.solve(coefficients)
        self.auxiliary = 2 * coupling.project(temporary)
        self.hidden_fe = 2 * temporary - coefficients

    def solve(self, rhs):
        return inverse_graph_mass(self.coupling, rhs, self.degree)

    def observe(self):
        return (
            2 * self.solve(self.coupling.adjoint(self.auxiliary) + self.hidden_fe)
            - self.hidden_fe
        )

    def energy(self):
        return self.coupling.energy(self.auxiliary) + self.coupling.space.energy(
            self.hidden_fe
        )

    def canonical_hidden(self):
        return self.coupling.project(self.value + self.hidden_fe) - self.auxiliary

    def advect(self, operation):
        new = operation(self.auxiliary)
        increment = 2 * self.solve(self.coupling.adjoint(new - self.auxiliary))
        self.auxiliary = new
        self.value = self.value + increment
        return self.value

    def apply_physics(self, operation, hidden_decay=1.0):
        if not 0 <= float(hidden_decay) <= 1:
            raise ValueError("Expected scalar hidden decay in [0,1]")
        new = operation(self.value)
        increment = new - hidden_decay * self.value
        temporary = self.solve(increment)
        self.auxiliary = hidden_decay * self.auxiliary + 2 * self.coupling.project(
            temporary
        )
        self.hidden_fe = hidden_decay * self.hidden_fe + 2 * temporary - increment
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
        temporary = self.solve(coupling.adjoint(moved) + self.value)
        self.auxiliary = 2 * coupling.project(temporary) - moved
        self.hidden_fe = 2 * temporary - self.value
        return dict(energy_loss=before - self.energy(), hidden_transfer_loss=loss)
