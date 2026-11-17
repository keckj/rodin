"""GPU moments of physical reaction applied to the complete ambient state.

Distributed under the Boost Software License, Version 1.0.
For q=a+d in V+W and g(x)=exp(-tau*(1+alpha*sin(2*pi*x))), evaluate
P_V(g*q) and P_W(g*q) through local matrices and weighted cross masses.
The resulting target is projected onto a small shared hidden space.
Nonpolynomial reaction weights require quadrature refinement checks.
"""

from gpu import cp, quadrature
from flux import gauss_interval
from carrier import Carrier


class ReactionMoments:
    def __init__(
        self,
        coupling,
        duration,
        amplitude=0.5,
        qdegree=20,
        diagnostics=False,
        subdivisions=1,
        increment=False,
        builder="auto",
        reanchor=False,
    ):
        self.coupling = coupling
        self.duration, self.amplitude = duration, amplitude
        self.diagnostics = diagnostics
        self.increment = increment
        self.reanchor = reanchor
        if reanchor and not increment:
            raise ValueError("Reanchoring here applies to increment moments")
        self.cross, self.fe, self.auxiliary = {}, {}, {}
        space = coupling.space
        q = quadrature(qdegree, subdivisions)
        basis = space.basis(q[:, :2])
        points = space.mesh.points(q)
        rates = 1 + amplitude * cp.sin(2 * cp.pi * points[..., 0])
        nodes, weights = gauss_interval(
            max(coupling.np + 5, qdegree // 2) * subdivisions
        )
        x = coupling.edges[0, :-1, None] + cp.diff(coupling.edges[0])[:, None] * nodes
        local = cp.ones((len(nodes), coupling.np))
        for i in range(coupling.np):
            for j in range(coupling.np):
                if i != j:
                    local[:, i] *= (nodes - coupling.nodes[j]) / (
                        coupling.nodes[i] - coupling.nodes[j]
                    )
        for power in (1, 2) if diagnostics else (1,):
            self.cross[power] = Carrier(
                space,
                n=coupling.n,
                degree=coupling.np - 1,
                qdegree=qdegree,
                edges=coupling.edges if coupling.nonuniform else None,
                reaction_weight=(
                    duration if increment else power * duration,
                    amplitude,
                ),
                subdivisions=subdivisions,
                reaction_form=(
                    ("increment" if power == 1 else "increment-square")
                    if increment
                    else "value"
                ),
                builder=builder,
            )
            weight = (
                cp.expm1(-duration * rates) ** power
                if increment
                else cp.exp(-power * duration * rates)
            )
            self.fe[power] = cp.ascontiguousarray(
                cp.einsum("cq,qi,qj,q->cij", weight, basis, basis, 2 * q[:, 2])
            )
            exponent = -duration * (1 + amplitude * cp.sin(2 * cp.pi * x))
            wx = cp.expm1(exponent) ** power if increment else cp.exp(power * exponent)
            self.auxiliary[power] = (
                cp.einsum("xq,qi,qj,q->xij", wx, local, local, weights)
                / coupling.weights[None, :, None]
            )
        self.decay_bound = cp.exp(cp.asarray(-duration * (1 - abs(amplitude))))

    def apply_fe(self, value, power=1):
        return cp.ascontiguousarray(cp.einsum("cij,cfj->cfi", self.fe[power], value))

    def apply_auxiliary(self, value, power=1):
        c = self.coupling
        reshaped = value.reshape(c.n, c.n, value.shape[1], c.np, c.np)
        result = cp.einsum("xij,yxfkj->yxfki", self.auxiliary[power], reshaped)
        return cp.ascontiguousarray(result.reshape(value.shape))

    def apply(
        self, state, iterations=2, shared=True, krylov_seed="moment", target_seed=False
    ):
        """Compress physical moments, optionally retaining their raw W moment.

        With target_seed, the retained ambient space contains P_W(f), so
        the projection error cannot exceed ||f-P_W(f)||. Incoming memory
        is also retained. This option requires at least two directions and
        applies only to full targets, not additive increment updates.
        """
        if target_seed and (
            iterations < 1 or self.increment or krylov_seed != "moment"
        ):
            raise ValueError(
                "Target seeding requires a full target and at least two moment directions"
            )
        if state.coupling is not self.coupling:
            raise ValueError("The weighted reaction belongs to a fixed support")
        c = self.coupling
        before = state.energy()
        previous = (state.value, state.anchor, state.auxiliary)
        a = state.anchor
        d = state.auxiliary - c.project(a)
        observation = self.apply_fe(a) + self.cross[1].adjoint(d)
        moment = self.cross[1].project(a) + self.apply_auxiliary(d)
        hidden = moment - c.project(observation)
        target_energy = None
        if self.diagnostics:
            mass = cp.broadcast_to(c.mass, (c.n**2, c.nc))
            target_energy = cp.einsum(
                "cfi,c,cfi->f", a, c.space.mesh.areas, self.apply_fe(a, 2)
            )
            target_energy += 2 * cp.einsum(
                "cfi,ci,cfi->f", self.cross[2].project(a), mass, d
            )
            target_energy += cp.einsum(
                "cfi,ci,cfi->f", d, mass, self.apply_auxiliary(d, 2)
            )
        search_seed = moment if target_seed else None
        if self.reanchor:
            search_seed = hidden
            if target_energy is not None:
                inner = cp.einsum("cfi,c,cfi->f", a, c.space.mesh.areas, observation)
                inner += cp.einsum("cfi,ci,cfi->f", d, mass, moment)
                target_energy = before + 2 * inner + target_energy
            observation = previous[0] + observation
            hidden = previous[2] - c.project(previous[0]) + hidden
        result = state.project_moments(
            c,
            observation,
            hidden,
            d,
            target_energy,
            iterations=iterations,
            shared=shared,
            gram_backend="reduction",
            krylov_seed=krylov_seed,
            search_seed=search_seed,
            incoming_diagnostic=self.increment,
        )
        if self.increment and not self.reanchor:
            # q belongs to V + span(H*d), hence P_X(q + delta) = q + P_X(delta).
            # The small solve acts on the physical increment. Its Pythagorean
            # defect is O(dt^2), avoiding subtraction of O(1) state energies.
            result["increment_energy"] = target_energy
            if self.diagnostics:
                result["increment_coefficients"] = tuple(
                    x.copy() for x in (state.value, state.anchor, state.auxiliary)
                )
            if target_energy is not None:
                inner = cp.einsum("cfi,c,cfi->f", a, c.space.mesh.areas, observation)
                inner += cp.einsum("cfi,ci,cfi->f", d, mass, moment)
                target_energy = before + 2 * inner + target_energy
            state.value += previous[0]
            state.anchor += previous[1]
            state.auxiliary += previous[2]
            result["energy"] = state.energy()
        result["target_energy"] = target_energy
        result["coercivity_excess"] = result["energy"] - self.decay_bound**2 * before
        return result
