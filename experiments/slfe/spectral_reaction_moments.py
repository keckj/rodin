"""Complete-state moments for symmetric multicomponent reaction on CUDA.

Distributed under the Boost Software License, Version 1.0.
The reaction is r(x) B with constant symmetric positive semidefinite B.
Diagonalization concerns the component matrix only. One shared hidden-space
projection keeps the result covariant under orthogonal component changes.
"""

from gpu import cp
from reaction_moments import ReactionMoments
from system_coupling_study import mix


class SpectralReactionMoments:
    def __init__(self, coupling, matrix, duration, amplitude=0.5, diagnostics=False):
        matrix = cp.asarray(matrix, dtype=cp.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("A square component matrix is required")
        if not bool(cp.allclose(matrix, matrix.T, atol=1e-13, rtol=1e-13)):
            raise ValueError("The component reaction must be symmetric")
        eigen, self.rotation = cp.linalg.eigh(matrix)
        tolerance = 1e-12 * max(1.0, float(cp.max(cp.abs(eigen))))
        if float(cp.min(eigen)) < -tolerance:
            raise ValueError("The component reaction must be positive semidefinite")
        self.rates = cp.where(eigen > tolerance, eigen, 0)
        self.coupling, self.diagnostics = coupling, diagnostics
        self.operators = []
        cache = {}
        for rate in self.rates.tolist():
            # Repeated component eigenvalues have the same spatial operator.
            # Rounding only merges eigensolver noise at the rank tolerance.
            key = round(rate, 12)
            if rate == 0:
                self.operators.append(None)
            else:
                if key not in cache:
                    cache[key] = ReactionMoments(
                        coupling, duration * rate, amplitude, diagnostics=diagnostics
                    )
                self.operators.append(cache[key])
        self.decay_bound = cp.exp(-duration * cp.min(self.rates) * (1 - abs(amplitude)))

    def apply(self, state, iterations=2, shared=True):
        if state.coupling is not self.coupling:
            raise ValueError("Reaction moments belong to a fixed support")
        c = self.coupling
        before = cp.sum(state.energy())
        d = state.auxiliary - c.project(state.anchor)
        a_modes, d_modes = mix(state.anchor, self.rotation), mix(d, self.rotation)
        v_modes = mix(state.value, self.rotation)
        w_modes = mix(state.auxiliary, self.rotation)
        observation, moment = cp.empty_like(v_modes), cp.empty_like(w_modes)
        total_target_energy = cp.asarray(0.0)
        mass = cp.broadcast_to(c.mass, (c.n**2, c.nc))
        for field, operator in enumerate(self.operators):
            a = cp.ascontiguousarray(a_modes[:, field : field + 1])
            detail = cp.ascontiguousarray(d_modes[:, field : field + 1])
            if operator is None:
                observation[:, field : field + 1] = v_modes[:, field : field + 1]
                moment[:, field : field + 1] = w_modes[:, field : field + 1]
                if self.diagnostics:
                    total_target_energy += cp.sum(c.space.energy(a))
                    total_target_energy += 2 * cp.einsum(
                        "cfi,ci,cfi->", c.project(a), mass, detail
                    )
                    total_target_energy += cp.sum(c.energy(detail))
            else:
                observation[:, field : field + 1] = operator.apply_fe(
                    a
                ) + operator.cross[1].adjoint(detail)
                moment[:, field : field + 1] = operator.cross[1].project(
                    a
                ) + operator.apply_auxiliary(detail)
                if self.diagnostics:
                    total_target_energy += cp.einsum(
                        "cfi,c,cfi->", a, c.space.mesh.areas, operator.apply_fe(a, 2)
                    )
                    total_target_energy += 2 * cp.einsum(
                        "cfi,ci,cfi->", operator.cross[2].project(a), mass, detail
                    )
                    total_target_energy += cp.einsum(
                        "cfi,ci,cfi->",
                        detail,
                        mass,
                        operator.apply_auxiliary(detail, 2),
                    )
        observation, moment = mix(observation, self.rotation.T), mix(
            moment, self.rotation.T
        )
        hidden = moment - c.project(observation)
        result = state.project_moments(
            c,
            observation,
            hidden,
            d,
            iterations=iterations,
            shared=shared,
            gram_backend="reduction",
        )
        after = cp.sum(result["energy"])
        result["total_coercivity_excess"] = after - self.decay_bound**2 * before
        if self.diagnostics:
            x = state.auxiliary - c.project(state.anchor)
            inner = cp.sum(c.space.energy(observation)) + cp.einsum(
                "cfi,ci,cfi->", hidden, mass, x
            )
            result["total_target_energy"] = total_target_energy
            result["total_projection_loss"] = total_target_energy - after
            result["total_distance_squared"] = total_target_energy + after - 2 * inner
        return result
