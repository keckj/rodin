"""A persistent auxiliary state with contractive FE physics updates.

Distributed under the Boost Software License, Version 1.0.
No square root or inverse of a projection defect is evaluated at runtime.
"""


class MemoryTransport:
    def __init__(self, coupling, coefficients):
        self.coupling = coupling
        self.value = coefficients.copy()
        self.auxiliary = coupling.project(coefficients)
        # z = (I-R*R)^(1/2) anchor is implicit throughout the evolution.
        self.anchor = coefficients.copy()

    def advect(self, operation):
        new = operation(self.auxiliary)
        self.value = self.value + self.coupling.adjoint(new - self.auxiliary)
        self.auxiliary = new
        return self.value

    def apply_physics(self, operation):
        new = operation(self.value)
        increment = new - self.value
        self.auxiliary = self.auxiliary + self.coupling.project(increment)
        self.anchor = self.anchor + increment
        self.value = new
        return self.value

    def energy(self):
        """Augmented energy without forming a projection-defect square root."""
        c = self.coupling
        return (
            c.energy(self.auxiliary)
            + c.space.energy(self.anchor)
            - c.energy(c.project(self.anchor))
        )

    def restart(self, coupling):
        """Change auxiliary space while preserving the current FE coefficients.

        The lifted transition is J_new J_old*. Its discarded squared norm is
        E_old - ||v||_V^2, a memory-loss diagnostic, not a PDE-error bound.
        Both couplings must act on the same FE space. Tiny negative losses
        from floating-point subtraction are returned without clipping.
        """
        if coupling.space is not self.coupling.space:
            raise ValueError("A memory restart requires the same FE space")
        loss = self.energy() - self.coupling.space.energy(self.value)
        self.coupling = coupling
        self.auxiliary = coupling.project(self.value)
        self.anchor = self.value.copy()
        return loss

    def transfer_memory(self, coupling, remap):
        """Carry the auxiliary part of hidden memory through a contraction.

        With r=w-R_old v, h=C r and b=R_new* h, set a_new=v-b and
        w_new=R_new a_new+h. The FE observation stays v. The new hidden
        energy is ||h||^2-||R_new* h||^2 and cannot exceed the old one.
        remap must contract the old auxiliary mass norm into the new one.
        """
        if coupling.space is not self.coupling.space:
            raise ValueError("A memory transfer requires the same FE space")
        before = self.energy()
        hidden = self.auxiliary - self.coupling.project(self.value)
        transported = remap(hidden)
        self.anchor = self.value - coupling.adjoint(transported)
        self.auxiliary = coupling.project(self.anchor) + transported
        self.coupling = coupling
        return before - self.energy()

    def refine(self, coupling, injection):
        """Reexpress the state on an exactly nested finer auxiliary space.

        injection must be an isometry C with C* R_new = R_old. Under these
        hypotheses a_new=a and w_new=C(w-R_old a)+R_new a preserve both the
        FE observation and the augmented energy. No root or inverse is used.
        The caller must establish nestedness; a general remap is insufficient.
        """
        if coupling.space is not self.coupling.space:
            raise ValueError("Auxiliary refinement requires the same FE space")
        before = self.energy()
        difference = self.auxiliary - self.coupling.project(self.anchor)
        self.auxiliary = injection(difference) + coupling.project(self.anchor)
        self.coupling = coupling
        return before - self.energy()

    def project_memory(self, coupling, cross_projection, iterations=1):
        """Project ambient hidden memory by a contractive polynomial filter.

        Both couplings must be exact L2 projections from the same FE space;
        cross_projection must equal P_Wnew restricted to Wold in that SAME
        ambient L2 space. An arbitrary contraction does not suffice here.
        Let h=P_Wnew(q-v), K=I-Rnew Rnew*. Starting at x=0, iterations of
        x <- h + Rnew Rnew* x give q_new=v+(I-P_V)x. Its hidden-space filter
        is 1-(1-lambda)^iterations, hence energy cannot increase and the FE
        observation is unchanged. The limit is the orthogonal projection
        onto V+Wnew; convergence may be slow for small principal angles.
        """
        if coupling.space is not self.coupling.space:
            raise ValueError("Memory projection requires the same FE space")
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("A positive integer iteration count is required")
        before = self.energy()
        h = cross_projection(
            self.auxiliary - self.coupling.project(self.anchor)
        ) + coupling.project(self.anchor - self.value)
        x = h.copy()
        for _ in range(iterations - 1):
            x = h + coupling.project(coupling.adjoint(x))
        self.anchor = self.value - coupling.adjoint(x)
        self.auxiliary = coupling.project(self.anchor) + x
        self.coupling = coupling
        return before - self.energy()
