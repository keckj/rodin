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

    def apply_physics(self, operation, hidden_decay=None):
        """Update FE physics, optionally damping the orthogonal hidden state.

        A scalar gamma in [0,1] gives q_new=G(v)+gamma*(q-v), hence
        E_new=||G(v)||^2+gamma^2*(E_old-||v||^2). The default leaves the
        hidden state unchanged. Uniform scalar reaction is represented
        exactly when G(v)=gamma*v; a generic FE operation does not determine
        a physically appropriate hidden decay by itself.
        """
        new = operation(self.value)
        if hidden_decay is None:
            increment = new - self.value
            self.auxiliary = self.auxiliary + self.coupling.project(increment)
            self.anchor = self.anchor + increment
        else:
            if not 0 <= float(hidden_decay) <= 1:
                raise ValueError("Hidden damping must lie between zero and one")
            increment = new - hidden_decay * self.value
            self.auxiliary = hidden_decay * self.auxiliary + self.coupling.project(
                increment
            )
            self.anchor = hidden_decay * self.anchor + increment
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

    def project_memory(
        self,
        coupling,
        cross_projection,
        iterations=1,
        filter_scale=None,
        diagnostics=None,
    ):
        """Project ambient hidden memory by a contractive polynomial filter.

        Both couplings must be exact L2 projections from the same FE space;
        cross_projection must equal P_Wnew restricted to Wold in that SAME
        ambient L2 space. An arbitrary contraction does not suffice here.
        Let h=P_Wnew(q-v), K=I-Rnew Rnew*. Starting at x=0, iterations of
        x <- h + Rnew Rnew* x give q_new=v+(I-P_V)x. Its hidden-space filter
        is 1-(1-lambda)^iterations, hence energy cannot increase and the FE
        observation is unchanged. The limit is the orthogonal projection
        onto V+Wnew; convergence may be slow for small principal angles.
        With 0 < filter_scale < 1, use a squared Chebyshev residual instead:
        g(lambda)=1-[T_m((1+s-2lambda)/(1-s))/T_m((1+s)/(1-s))]^2.
        This is contractive for EVERY lambda in [0,1], including below s;
        s is a target scale, not an assumed spectral lower bound. Its degree
        is 2*iterations and costs 2*iterations-1 applications of Rnew Rnew*.
        If supplied, diagnostics receives per-field squared ambient change
        and energy loss. For these positive contractive filters the former
        is bounded by the latter. This bounds remapping, not full PDE error.
        """
        if coupling.space is not self.coupling.space:
            raise ValueError("Memory projection requires the same FE space")
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("A positive integer iteration count is required")
        if filter_scale is not None and not 0 < filter_scale < 1:
            raise ValueError("The Chebyshev target scale must belong to (0,1)")
        before = self.energy()
        h = cross_projection(
            self.auxiliary - self.coupling.project(self.anchor)
        ) + coupling.project(self.anchor - self.value)
        if filter_scale is None:
            x = h.copy()
            for _ in range(iterations - 1):
                x = h + coupling.project(coupling.adjoint(x))
        else:
            a = (1 + filter_scale) / (1 - filter_scale)
            b = 2 / (1 - filter_scale)

            def defect(z):
                return z - coupling.project(coupling.adjoint(z))

            def chebyshev(rhs):
                # Stable three-term recurrence for (1-r_m(K))/K.
                previous = 0 * rhs
                current = (b / a) * rhs
                ratio = 1 / a
                for _ in range(1, iterations):
                    next_ratio = 1 / (2 * a - ratio)
                    following = (
                        2 * a * next_ratio * current
                        - ratio * next_ratio * previous
                        + 2 * b * next_ratio * (rhs - defect(current))
                    )
                    previous, current, ratio = current, following, next_ratio
                return current

            first = chebyshev(h)
            x = first + chebyshev(h - defect(first))
        self.anchor = self.value - coupling.adjoint(x)
        self.auxiliary = coupling.project(self.anchor) + x
        self.coupling = coupling
        after = self.energy()
        loss = before - after
        if diagnostics is not None:
            # <h,x> by polarization in Wnew, independent of its mass layout.
            cross = (coupling.energy(h + x) - coupling.energy(h - x)) / 4
            diagnostics.update(
                energy_loss=loss,
                ambient_change_squared=(
                    before + after - 2 * coupling.space.energy(self.value) - 2 * cross
                ),
            )
        return loss

    def project_memory_warm(self, coupling, cross_projection, iterations=1):
        """Experimental identity-consistent transfer with an energy limiter.

        Exact common-ambient projections and DG layout (cell, field, node)
        are required. Transport the old auxiliary representative as a warm
        start, correct its normal-equation residual, then project the new
        hidden state radially into the old hidden-energy ball. This preserves
        v and its energy bound and is the identity when the spaces coincide.
        The state-dependent limiter makes this transfer nonlinear; an energy
        bound does not imply pairwise nonexpansiveness or an error estimate.
        """
        if coupling.space is not self.coupling.space:
            raise ValueError("Memory projection requires the same FE space")
        if not isinstance(iterations, int) or iterations < 0:
            raise ValueError("A nonnegative integer correction count is required")
        before = self.energy()
        visible = coupling.space.energy(self.value)
        old_hidden = (before - visible).clip(0)
        d = self.auxiliary - self.coupling.project(self.anchor)
        x = cross_projection(d)
        h = x + coupling.project(self.anchor - self.value)
        for _ in range(iterations):
            x = h + coupling.project(coupling.adjoint(x))
        b = coupling.adjoint(x)
        proposed_hidden = (coupling.energy(x) - coupling.space.energy(b)).clip(0)
        scale = (old_hidden / proposed_hidden.clip(1e-300)).clip(0, 1) ** 0.5
        x = x * scale[None, :, None]
        self.anchor = self.value - b * scale[None, :, None]
        self.auxiliary = coupling.project(self.anchor) + x
        self.coupling = coupling
        after = self.energy()
        cross = (coupling.energy(h + x) - coupling.energy(h - x)) / 4
        return dict(
            energy_loss=before - after,
            ambient_change_squared=before + after - 2 * visible - 2 * cross,
            limiter=scale,
        )

    def project_memory_galerkin(
        self,
        coupling,
        cross_projection,
        iterations=2,
        shared=False,
        gram_backend="einsum",
        krylov_seed="moment",
    ):
        """Project onto a small augmented hidden subspace, using CUDA Gram solves.

        This experimental DG-only path augments a Krylov subspace by the
        transferred old representative. Orthogonal projection in its ambient
        image preserves the FE field and bounds energy at every subspace
        size. With an unchanged support, the warm direction already spans
        the old hidden state, subject to the numerical rank threshold. The
        subspace depends on the field, so this is nonlinear. Tiny Gram
        eigenvalues are discarded at a stated tolerance.

        With shared=True all fields use the same augmented ambient space.
        This preserves orthogonal changes of field coordinates and gives a
        positive-semidefinite loss of the full field Gram matrix. Direction
        normalization uses one scalar per block to retain this covariance.
        """
        if coupling.space is not self.coupling.space:
            raise ValueError("Memory projection requires the same FE space")
        if not isinstance(iterations, int) or iterations < 0:
            raise ValueError("A nonnegative Krylov size is required")
        if gram_backend not in ("einsum", "reduction"):
            raise ValueError("Expected einsum or reduction Gram assembly")
        before = self.energy()
        d = self.auxiliary - self.coupling.project(self.anchor)
        mapped = cross_projection(d)
        h = mapped + coupling.project(self.anchor - self.value)
        return self.project_moments(
            coupling,
            self.value,
            h,
            mapped,
            before,
            iterations=iterations,
            shared=shared,
            gram_backend=gram_backend,
            krylov_seed=krylov_seed,
        )

    def project_moments(
        self,
        coupling,
        observation,
        hidden_moment,
        incoming_direction,
        target_energy=None,
        iterations=2,
        shared=False,
        gram_backend="einsum",
        krylov_seed="moment",
        search_seed=None,
        incoming_diagnostic=False,
    ):
        """Project ambient moments onto V plus a small hidden subspace.

        The caller supplies P_V(q), P_W(q-P_V(q)), and a direction in W.
        These moments must describe the same ambient target q. If supplied,
        target_energy is its squared norm and enables Pythagorean diagnostics.
        This interface also permits physical multiplication of the complete
        ambient state before projection, including the hidden memory.
        """
        from gpu import cp

        if coupling.space is not self.coupling.space:
            raise ValueError("Moment projection requires the same FE space")
        if not isinstance(iterations, int) or iterations < 0:
            raise ValueError("A nonnegative Krylov size is required")
        if gram_backend not in ("einsum", "reduction"):
            raise ValueError("Expected einsum or reduction Gram assembly")
        if krylov_seed not in ("moment", "increment", "residual"):
            raise ValueError("Expected moment, increment or residual Krylov seed")
        before = target_energy
        self.value = observation
        visible = coupling.space.energy(observation)
        h, mapped = hidden_moment, incoming_direction
        mass = cp.broadcast_to(coupling.mass, (coupling.n**2, coupling.nc))
        vectors, images = [], []
        candidate = mapped
        for index in range(iterations + 1):
            energy = coupling.energy(candidate)
            norm = cp.sqrt(cp.sum(energy) if shared else energy).clip(1e-150)
            vector = candidate / (norm if shared else norm[None, :, None])
            image = coupling.adjoint(vector)
            vectors.append(vector)
            images.append(image)
            if index == 0:
                incoming_norm = norm
                candidate = h if search_seed is None else search_seed
                if krylov_seed != "moment" and iterations:
                    scale = norm if shared else norm[None, :, None]
                    kmapped = mapped - coupling.project(image * scale)
                    factor = 1.0
                    if krylov_seed == "residual":
                        denominator = cp.sum(
                            mapped * kmapped * mass[:, None, :], axis=(0, 2)
                        )
                        numerator = cp.sum(mapped * h * mass[:, None, :], axis=(0, 2))
                        reference = energy
                        if shared:
                            denominator, numerator, reference = map(
                                cp.sum, (denominator, numerator, reference)
                            )
                        valid = denominator > 1e-12 * reference
                        factor = cp.where(
                            valid, numerator / cp.where(valid, denominator, 1), 0
                        )
                        if not shared:
                            factor = factor[None, :, None]
                    candidate = h - factor * kmapped
            else:
                candidate = coupling.project(image)
        xbasis = cp.stack(vectors, axis=-1)
        vbasis = cp.stack(images, axis=-1)
        if gram_backend == "reduction":
            from weighted_products import weighted_products

            if shared:
                xbasis = xbasis.transpose(0, 2, 3, 1).reshape(
                    coupling.n**2, coupling.nc, -1
                )
                vbasis = vbasis.transpose(0, 2, 3, 1).reshape(
                    coupling.space.mesh.cells, coupling.space.nb, -1
                )
                xflat = xbasis.reshape(1, -1, xbasis.shape[-1])
                vflat = vbasis.reshape(1, -1, vbasis.shape[-1])
                hflat = h.transpose(0, 2, 1).reshape(1, -1, h.shape[1])
            else:
                xflat = xbasis.transpose(1, 0, 2, 3).reshape(
                    h.shape[1], -1, iterations + 1
                )
                vflat = vbasis.transpose(1, 0, 2, 3).reshape(
                    h.shape[1], -1, iterations + 1
                )
                hflat = h.transpose(1, 0, 2).reshape(h.shape[1], -1, 1)
            gram = weighted_products(xflat, xflat, mass.ravel())
            gram -= weighted_products(
                vflat, vflat, cp.repeat(coupling.space.mesh.areas, coupling.space.nb)
            )
            rhs = weighted_products(xflat, hflat, mass.ravel())
            if shared:
                gram, rhs = gram[0], rhs[0]
                gram = (gram + gram.T) / 2
            else:
                gram = (gram + gram.transpose(0, 2, 1)) / 2
                rhs = rhs[..., 0]
        elif shared:
            fields = h.shape[1]
            xbasis = xbasis.transpose(0, 2, 3, 1).reshape(
                coupling.n**2, coupling.nc, -1
            )
            vbasis = vbasis.transpose(0, 2, 3, 1).reshape(
                coupling.space.mesh.cells, coupling.space.nb, -1
            )
            gram = cp.einsum("cik,ci,cil->kl", xbasis, mass, xbasis)
            gram -= cp.einsum(
                "cik,c,cil->kl", vbasis, coupling.space.mesh.areas, vbasis
            )
            gram = (gram + gram.T) / 2
            rhs = cp.einsum("cfi,ci,cik->kf", h, mass, xbasis)
            assert rhs.shape == ((iterations + 1) * fields, fields)
        else:
            gram = cp.einsum("cfik,ci,cfil->fkl", xbasis, mass, xbasis)
            gram -= cp.einsum(
                "cfik,c,cfil->fkl", vbasis, coupling.space.mesh.areas, vbasis
            )
            gram = (gram + gram.transpose(0, 2, 1)) / 2
            rhs = cp.einsum("cfi,ci,cfik->fk", h, mass, xbasis)
        eigen, eigenvectors = cp.linalg.eigh(gram)
        assert float(eigen.min()) > -1e-8, float(eigen.min())
        # Mass-normalized directions make an absolute cutoff meaningful.
        # The relative part protects nearly dependent augmented directions.
        cutoff = 1e-12 * cp.maximum(eigen[-1] if shared else eigen[:, -1:], 1.0)
        keep = eigen > cutoff
        inverse = cp.where(keep, 1 / cp.maximum(eigen, cutoff), 0)
        if shared:
            coefficients = eigenvectors @ (inverse[:, None] * (eigenvectors.T @ rhs))
            x = cp.einsum("cik,kf->cfi", xbasis, coefficients)
            b = cp.einsum("cik,kf->cfi", vbasis, coefficients)
        else:
            coordinates = cp.einsum("fij,fi->fj", eigenvectors, rhs)
            coefficients = cp.einsum("fij,fj->fi", eigenvectors, inverse * coordinates)
            x = cp.einsum("cfik,fk->cfi", xbasis, coefficients)
            b = cp.einsum("cfik,fk->cfi", vbasis, coefficients)
        self.anchor = self.value - b
        self.auxiliary = coupling.project(self.anchor) + x
        self.coupling = coupling
        after = self.energy()
        cross = cp.einsum("cfi,ci,cfi->f", h, mass, x)
        result = dict(
            energy=after,
            energy_loss=None if before is None else before - after,
            ambient_change_squared=(
                None if before is None else before + after - 2 * visible - 2 * cross
            ),
            gram_eigenvalues=eigen,
            retained_rank=cp.sum(keep, axis=-1),
        )
        if incoming_diagnostic:
            dropped = cp.where(keep, 0, cp.maximum(eigen, 0))
            if shared:
                loss = cp.sum(
                    eigenvectors[: h.shape[1]] ** 2 * dropped[None, :], axis=1
                )
            else:
                loss = cp.sum(eigenvectors[:, 0, :] ** 2 * dropped, axis=1)
            result["incoming_projection_loss"] = incoming_norm**2 * loss
        return result
