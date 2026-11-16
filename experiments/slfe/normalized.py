"""Positive scalar coupling and constant-anchored local Gram normalization.

Distributed under the Boost Software License, Version 1.0.
This experiment changes the sampled high modes rather than requiring exact
degree-2p cubature on both marginals. Accuracy must be measured separately.
"""
import time

from gpu import cp, quadrature
from moments import MomentTransfer


def layered_particles(space, flow, t0, t1, degree=8, thickness=2.):
    """Partition each reference triangle into a core and six boundary triangles.

    The core's barycentric offsets follow the outgoing vertex displacement.
    Layers shrink with the time step, keeping quadrature sites near crossing
    faces without intersections or a time-step-dependent particle count.
    The positive source quadrature remains polynomial-exact through degree.
    """
    mesh = space.mesh
    q = quadrature(max(degree, 2*space.degree))
    t = mesh.triangles
    delta = flow(t, t0, t1)-t
    e, f = t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]
    det = 2*mesh.areas
    dr = (delta[..., 0]*f[:, None, 1]-delta[..., 1]*f[:, None, 0])/det[:, None]
    ds = (e[:, None, 0]*delta[..., 1]-e[:, None, 1]*delta[..., 0])/det[:, None]
    dlambda = cp.stack((-dr-ds, dr, ds), axis=-1)
    offset = thickness*cp.maximum(-cp.min(dlambda, axis=1), 0)
    # Keep a nondegenerate core for large moves; the adaptive-layer claim
    # concerns the small-displacement regime, not large-CFL accuracy.
    offset *= cp.minimum(1., .5/cp.maximum(cp.sum(offset, axis=1), 1e-300))[:, None]
    core = offset[:, None, :]+(1-cp.sum(offset, axis=1))[:, None, None]*cp.eye(3)[None]
    inner = core[:, :, 1:]
    outer = cp.broadcast_to(cp.asarray([[0., 0.], [1., 0.], [0., 1.]]), inner.shape)
    triangles = [inner]
    for i in range(3):
        j = (i+1) % 3
        triangles.append(cp.stack((outer[:, i], outer[:, j], inner[:, j]), axis=1))
        triangles.append(cp.stack((outer[:, i], inner[:, j], inner[:, i]), axis=1))
    triangles = cp.stack(triangles, axis=1)
    a = triangles[:, :, 1]-triangles[:, :, 0]
    b = triangles[:, :, 2]-triangles[:, :, 0]
    determinants = a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    points = triangles[:, :, None, 0]+q[None, None, :, :1]*a[:, :, None]+q[None, None, :, 1:2]*b[:, :, None]
    weights = 2*mesh.areas[:, None, None]*determinants[:, :, None]*q[None, None, :, 2]
    source = cp.broadcast_to(cp.arange(mesh.cells, dtype=cp.int32)[:, None, None], weights.shape)
    keep = weights.ravel() > 0
    return (cp.ascontiguousarray(source.ravel()[keep]), cp.ascontiguousarray(points.reshape(-1, 2)[keep]),
            cp.ascontiguousarray(weights.ravel()[keep]))


class NormalizedTransfer(MomentTransfer):
    """Balance degree zero, then orthonormalize the remaining modes per cell.

    If both mass marginals are exact, the resulting weighted evaluations are
    isometries with a shared representation of the constant. Their cross
    product conserves mass and constants and is L2 contractive. This is not
    an assertion that the original polynomial mixed moments are exact.
    """
    def __init__(self, space, flow, t0, t1, support="layers", qdegree=8,
                 subdivisions=1, solver="cg", tolerance=1e-11, maxiter=4000):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        particles = layered_particles(space, flow, t0, t1, degree=qdegree) if support == "layers" else None
        if support not in ("layers", "interior"):
            raise ValueError(support)
        super().__init__(space, flow, t0, t1, qdegree=qdegree, subdivisions=subdivisions,
                         fit="positive", solver=solver, moment_degree=0,
                         tolerance=tolerance, maxiter=maxiter, particle_data=particles)
        self.support = support
        self.balance_seconds = self.build_seconds
        if self.relative_moment_residual > 1e-8 or self.minimum_weight_ratio < -1e-12:
            raise RuntimeError(f"Scalar mass coupling failed: {self.summary()}")
        self.gram_defects = []
        self.minimum_covariance_eigenvalues = []
        self.sb = self.normalize(self.source, self.sb)
        self.tb = self.normalize(self.target, self.tb)
        cp.cuda.Stream.null.synchronize()
        self.build_seconds = time.perf_counter()-start

    def normalize(self, cells, basis):
        mesh = self.space.mesh
        nb = self.space.nb
        gram = cp.empty((mesh.cells, nb, nb), dtype=cp.float64)
        for i in range(nb):
            for j in range(i+1):
                g = cp.bincount(cells, weights=self.weights*basis[:, i]*basis[:, j], minlength=mesh.cells)/mesh.areas
                gram[:, i, j] = g
                gram[:, j, i] = g
        if nb == 1:
            return basis
        mean = gram[:, 0, 1:]/gram[:, 0, 0, None]
        covariance = gram[:, 1:, 1:]-gram[:, 0, 0, None, None]*mean[:, :, None]*mean[:, None, :]
        values, vectors = cp.linalg.eigh(covariance)
        minimum = float(cp.min(values))
        if minimum <= 1e-12:
            raise RuntimeError(f"Rank-deficient local sampling covariance: {minimum}")
        inverse_root = (vectors/cp.sqrt(values)[:, None, :])@vectors.transpose(0, 2, 1)
        normalized = cp.empty_like(basis)
        normalized[:, 0] = 1.
        normalized[:, 1:] = cp.einsum("li,lij->lj", basis[:, 1:]-mean[cells], inverse_root[cells])
        identity = cp.eye(nb-1)[None]
        defect = float(cp.max(cp.abs(inverse_root@covariance@inverse_root-identity)))
        self.minimum_covariance_eigenvalues.append(minimum)
        self.gram_defects.append(defect)
        return cp.ascontiguousarray(normalized)

    def summary(self):
        result = super().summary()
        result.update(method="scalar_balance_local_gram", support=getattr(self, "support", None),
                      gram_defects=getattr(self, "gram_defects", None),
                      minimum_covariance_eigenvalues=getattr(self, "minimum_covariance_eigenvalues", None))
        return result
