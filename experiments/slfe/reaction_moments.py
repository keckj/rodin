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
    ):
        self.coupling = coupling
        self.duration, self.amplitude = duration, amplitude
        self.diagnostics = diagnostics
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
                reaction_weight=(power * duration, amplitude),
                subdivisions=subdivisions,
            )
            weight = cp.exp(-power * duration * rates)
            self.fe[power] = cp.ascontiguousarray(
                cp.einsum("cq,qi,qj,q->cij", weight, basis, basis, 2 * q[:, 2])
            )
            wx = cp.exp(-power * duration * (1 + amplitude * cp.sin(2 * cp.pi * x)))
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

    def apply(self, state, iterations=2, shared=True):
        if state.coupling is not self.coupling:
            raise ValueError("The weighted reaction belongs to a fixed support")
        c = self.coupling
        before = state.energy()
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
        result = state.project_moments(
            c,
            observation,
            hidden,
            d,
            target_energy,
            iterations=iterations,
            shared=shared,
            gram_backend="reduction",
        )
        result["target_energy"] = target_energy
        result["coercivity_excess"] = result["energy"] - self.decay_bound**2 * before
        return result
