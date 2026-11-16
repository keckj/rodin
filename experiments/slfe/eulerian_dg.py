"""GPU central-flux DG/RK4 baseline on the same jittered triangular spaces.

Distributed under the Boost Software License, Version 1.0.
The prescribed velocity is the autonomous cellular field used in day_study.
"""

from gpu import cp, HERE, quadrature, launch
from flux import gauss_interval
from flows import cellular_velocity


class EulerianDG:
    def __init__(self, space, degree=18, face_order=12, flux="central"):
        if flux not in ("central", "upwind"):
            raise ValueError("Flux must be central or upwind")
        self.space = space
        mesh = space.mesh
        code = f"#define NB {space.nb}\n#define M_SQRT2 1.41421356237309504880\n"
        code += (HERE / "basis_gradient.cuh").read_text() + (
            HERE / "eulerian_dg.cu"
        ).read_text()
        self.module = cp.RawModule(code=code)
        self.apply_kernel = self.module.get_function("dg_rhs")
        q = quadrature(degree)
        b = space.basis(q[:, :2])
        gradients = cp.empty((len(q), space.nb, 2), dtype=cp.float64)
        launch(
            self.module.get_function("evaluate_gradients"),
            len(q),
            (cp.ascontiguousarray(q[:, :2]), gradients, len(q)),
        )
        e = mesh.triangles[:, 1] - mesh.triangles[:, 0]
        f = mesh.triangles[:, 2] - mesh.triangles[:, 0]
        inverse = cp.linalg.inv(cp.stack((e, f), axis=-1))
        physical = cp.einsum("qia,cab->cqib", gradients, inverse)
        velocity = cellular_velocity(mesh.points(q), 0)
        directional = cp.einsum("cqib,cqb->cqi", physical, velocity)
        volume = cp.einsum("cqi,qj,q->cij", directional, b, 2 * q[:, 2])
        # Recover periodic vertex IDs from this benchmark mesh's lattice
        # ancestry. All topology and sorting work stays on the GPU.
        if hasattr(mesh, "vertex_ids"):
            ids = mesh.vertex_ids
        else:
            vertex = cp.rint(mesh.triangles * mesh.n).astype(cp.int64) % mesh.n
            ids = vertex[:, :, 1] * mesh.n + vertex[:, :, 0]
        following = cp.roll(ids, -1, axis=1)
        keys = cp.minimum(ids, following) * mesh.n**2 + cp.maximum(ids, following)
        order = cp.argsort(keys.ravel())
        paired = keys.ravel()[order]
        assert bool(cp.all(paired[::2] == paired[1::2]))
        assert bool(cp.all(paired[1:-1:2] != paired[2::2]))
        mates = cp.empty(mesh.cells * 3, dtype=cp.int32)
        mates[order[::2]] = order[1::2].astype(cp.int32)
        mates[order[1::2]] = order[::2].astype(cp.int32)
        self.neighbor = cp.ascontiguousarray((mates // 3).reshape(mesh.cells, 3))
        nodes, weights = gauss_interval(face_order)
        corners = cp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        ref = (
            corners[:, None, :]
            + nodes[None, :, None]
            * (cp.roll(corners, -1, axis=0) - corners)[:, None, :]
        )
        trace = space.basis(ref).reshape(3, face_order, space.nb)
        neighbor_trace = trace[(mates % 3).reshape(mesh.cells, 3), ::-1, :]
        starts = mesh.triangles
        edges = cp.roll(starts, -1, axis=1) - starts
        positions = (
            starts[:, :, None, :] + nodes[None, None, :, None] * edges[:, :, None, :]
        )
        scaled_normal = cp.stack((edges[:, :, 1], -edges[:, :, 0]), axis=-1)
        speed = cp.einsum(
            "ceqd,ced->ceq", cellular_velocity(positions, 0), scaled_normal
        )
        self_face = cp.einsum("ceq,eqi,eqj,q->cij", speed, trace, trace, weights)
        diagonal = volume - 0.5 * self_face / mesh.areas[:, None, None]
        offdiagonal = (
            -0.5
            * cp.einsum("ceq,eqi,ceqj,q->ceij", speed, trace, neighbor_trace, weights)
            / mesh.areas[:, None, None, None]
        )
        # Explicitly remove only the quadrature/roundoff symmetric defect.
        # The size of this correction is recorded and independently refined.
        corrected_diagonal = 0.5 * (diagonal - diagonal.transpose(0, 2, 1))
        other = (
            offdiagonal.reshape(-1, space.nb, space.nb)[mates]
            .transpose(0, 2, 1)
            .reshape(offdiagonal.shape)
        )
        ratio = mesh.areas[self.neighbor] / mesh.areas[:, None]
        corrected_offdiagonal = 0.5 * (offdiagonal - other * ratio[:, :, None, None])
        scale = max(float(cp.max(cp.abs(diagonal))), float(cp.max(cp.abs(offdiagonal))))
        self.skew_correction = max(
            float(cp.max(cp.abs(corrected_diagonal - diagonal))),
            float(cp.max(cp.abs(corrected_offdiagonal - offdiagonal))),
        ) / max(1.0, scale)
        assert self.skew_correction < 1e-9, self.skew_correction
        self.diagonal = cp.ascontiguousarray(corrected_diagonal)
        self.offdiagonal = cp.ascontiguousarray(corrected_offdiagonal)
        if flux == "upwind":
            self.diagonal -= (
                0.5
                * cp.einsum("ceq,eqi,eqj,q->cij", cp.abs(speed), trace, trace, weights)
                / mesh.areas[:, None, None]
            )
            self.offdiagonal += (
                0.5
                * cp.einsum(
                    "ceq,eqi,ceqj,q->ceij",
                    cp.abs(speed),
                    trace,
                    neighbor_trace,
                    weights,
                )
                / mesh.areas[:, None, None, None]
            )
        row_sum = cp.sum(cp.abs(self.diagonal), axis=-1)
        row_sum += cp.sum(
            cp.abs(self.offdiagonal) / cp.sqrt(ratio)[:, :, None, None], axis=(1, 3)
        )
        self.norm_bound = float(cp.max(row_sum))

    def rhs(self, value):
        result = cp.empty_like(value)
        launch(
            self.apply_kernel,
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

    def step(self, value, dt):
        a = self.rhs(value)
        b = self.rhs(value + 0.5 * dt * a)
        c = self.rhs(value + 0.5 * dt * b)
        d = self.rhs(value + dt * c)
        return value + (dt / 6) * (a + 2 * b + 2 * c + d)
