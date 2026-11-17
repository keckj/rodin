"""GPU SIPG diffusion and mass-scaled Crank--Nicolson solves.

Distributed under the Boost Software License, Version 1.0.
Geometry and polynomial bases match the existing jittered triangular DG
prototype. The local block action reuses its CUDA kernel; no CPU solve occurs.
"""

from cupyx.scipy.sparse.linalg import LinearOperator, cg

from gpu import cp, HERE, quadrature, launch
from flux import gauss_interval
from eulerian_dg import periodic_mates


class FEDiffusion:
    def __init__(self, space, diffusivity=0.002, penalty=8.0):
        if diffusivity < 0 or penalty <= 0:
            raise ValueError("Expected nonnegative diffusivity and positive penalty")
        self.space, self.diffusivity = space, diffusivity
        mesh, nb = space.mesh, space.nb
        code = f"#define NB {nb}\n#define M_SQRT2 1.41421356237309504880\n"
        code += (HERE / "basis_gradient.cuh").read_text() + (
            HERE / "eulerian_dg.cu"
        ).read_text()
        self.module = cp.RawModule(code=code)
        self.kernel = self.module.get_function("dg_rhs")
        gradient_kernel = self.module.get_function("evaluate_gradients")
        e = mesh.triangles[:, 1] - mesh.triangles[:, 0]
        f = mesh.triangles[:, 2] - mesh.triangles[:, 0]
        inverse = cp.linalg.inv(cp.stack((e, f), axis=-1))
        q = quadrature(max(1, 2 * space.degree))
        gradient = cp.empty((len(q), nb, 2))
        launch(
            gradient_kernel, len(q), (cp.ascontiguousarray(q[:, :2]), gradient, len(q))
        )
        physical = cp.einsum("qia,cab->cqib", gradient, inverse)
        diagonal = cp.einsum(
            "cqid,cqjd,q,c->cij", physical, physical, 2 * q[:, 2], mesh.areas
        )
        mates = periodic_mates(mesh)
        self.neighbor = cp.ascontiguousarray((mates // 3).reshape(mesh.cells, 3))
        nodes, weights = gauss_interval(space.degree + 2)
        corners = cp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        ref = (
            corners[:, None, :]
            + nodes[None, :, None]
            * (cp.roll(corners, -1, axis=0) - corners)[:, None, :]
        )
        trace = space.basis(ref).reshape(3, len(nodes), nb)
        other_trace = trace[(mates % 3).reshape(mesh.cells, 3), ::-1, :]
        gradient = cp.empty((ref.shape[0] * ref.shape[1], nb, 2))
        launch(
            gradient_kernel,
            len(gradient),
            (cp.ascontiguousarray(ref.reshape(-1, 2)), gradient, len(gradient)),
        )
        gradient = gradient.reshape(3, len(nodes), nb, 2)
        physical = cp.einsum("eqia,cab->ceqib", gradient, inverse)
        other_gradient = physical.reshape(-1, len(nodes), nb, 2)[mates, ::-1].reshape(
            physical.shape
        )
        edges = cp.roll(mesh.triangles, -1, axis=1) - mesh.triangles
        length = cp.sqrt(cp.sum(edges**2, axis=2))
        normal = (
            cp.stack((edges[:, :, 1], -edges[:, :, 0]), axis=-1) / length[..., None]
        )
        dn = cp.einsum("ceqib,ceb->ceqi", physical, normal)
        other_dn = cp.einsum("ceqib,ceb->ceqi", other_gradient, normal)
        inverse_height = length / (2 * mesh.areas[:, None])
        other_height = inverse_height.ravel()[mates].reshape(mesh.cells, 3)
        # Conservative fixed penalty for this affine jittered-mesh experiment.
        # Coercivity is independently checked; this is not an arbitrary-mesh bound.
        sigma = (
            penalty * (space.degree + 1) ** 2 * cp.maximum(inverse_height, other_height)
        )
        diagonal += (
            -0.5 * cp.einsum("eqi,ceqj,q,ce->cij", trace, dn, weights, length)
            - 0.5 * cp.einsum("ceqi,eqj,q,ce->cij", dn, trace, weights, length)
            + cp.einsum("eqi,eqj,q,ce->cij", trace, trace, weights, length * sigma)
        )
        off = (
            -0.5 * cp.einsum("eqi,ceqj,q,ce->ceij", trace, other_dn, weights, length)
            + 0.5 * cp.einsum("ceqi,ceqj,q,ce->ceij", dn, other_trace, weights, length)
            - cp.einsum(
                "eqi,ceqj,q,ce->ceij", trace, other_trace, weights, length * sigma
            )
        )
        self.root_mass = cp.sqrt(mesh.areas)[:, None, None]
        self.diagonal = cp.ascontiguousarray(diagonal / mesh.areas[:, None, None])
        self.offdiagonal = cp.ascontiguousarray(
            off
            / cp.sqrt(mesh.areas[:, None] * mesh.areas[self.neighbor])[:, :, None, None]
        )
        self.solve_history = []
        self.preconditioners = {}

    def apply_scaled(self, value):
        value = cp.ascontiguousarray(value)
        result = cp.empty_like(value)
        launch(
            self.kernel,
            value.size,
            (
                self.diagonal,
                self.offdiagonal,
                self.neighbor,
                value,
                result,
                self.space.mesh.cells,
                value.shape[1],
            ),
        )
        return result

    def step(self, coefficients, duration):
        """CN update with a reused block operator and Jacobi preconditioner."""
        if duration < 0:
            raise ValueError("Diffusion duration must be nonnegative")
        factor = 0.5 * duration * self.diffusivity
        cells, fields, nb = coefficients.shape
        size = cells * nb
        if factor not in self.preconditioners:
            diagonal = (
                1 + factor * cp.diagonal(self.diagonal, axis1=1, axis2=2)
            ).ravel()
            self.preconditioners[factor] = LinearOperator(
                (size, size), matvec=lambda x, d=diagonal: x / d, dtype=cp.float64
            )
        apply = lambda x: self.apply_scaled(x.reshape(cells, 1, nb)).ravel()
        operator = LinearOperator(
            (size, size), matvec=lambda x: x + factor * apply(x), dtype=cp.float64
        )
        result = cp.empty_like(coefficients)
        for field in range(fields):
            initial = cp.ascontiguousarray(
                coefficients[:, field : field + 1] * self.root_mass
            ).ravel()
            rhs = initial - factor * apply(initial)
            iterations = [0]

            def count(_):
                iterations[0] += 1

            value, info = cg(
                operator,
                rhs,
                x0=initial,
                M=self.preconditioners[factor],
                tol=2e-12,
                atol=0,
                maxiter=1000,
                callback=count,
            )
            if info:
                raise RuntimeError(f"Diffusion CG failed: {info}")
            residual = float(
                cp.linalg.norm(operator @ value - rhs)
                / cp.maximum(cp.linalg.norm(rhs), 1e-100)
            )
            assert residual < 1e-10, residual
            self.solve_history.append(
                dict(iterations=iterations[0], relative_residual=residual)
            )
            result[:, field : field + 1] = value.reshape(cells, 1, nb) / self.root_mass
        return result
