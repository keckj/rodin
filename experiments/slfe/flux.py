"""Characteristic face-flux means with backward SL projection of higher modes.

Distributed under the Boost Software License, Version 1.0.
GPU research baseline: conservation follows from shared oriented face fluxes.
No claim of unconditional energy stability is made by this construction.
"""
import time
from functools import lru_cache

from gpu import cp, quadrature, launch, HERE


@lru_cache(None)
def gauss_interval(order):
    k = cp.arange(1, order, dtype=cp.float64)
    b = k/cp.sqrt(4*k*k-1)
    jacobi = cp.diag(b, 1)+cp.diag(b, -1)
    x, vectors = cp.linalg.eigh(jacobi)
    return (x+1)/2, vectors[0]**2


FLUX_KERNEL = r'''
extern "C" __global__ void flux_means(const int* source, const int* target,
 const double* basis, const double* weights, const double* areas,
 int count, const double* input, double* output, int fields, int nb)
{
  int item = blockIdx.x*blockDim.x+threadIdx.x;
  if (item >= count*fields) return;
  int point = item/fields, field = item%fields;
  double value = 0.;
  for (int k=0; k<nb; ++k)
    value += basis[point*nb+k]*input[(source[point]*fields+field)*nb+k];
  atomicAdd(output+(target[point]*fields+field)*nb,
            -weights[point]*value/areas[target[point]]);
}
'''


class FluxTransfer:
    def __init__(self, space, flow, velocity, t0, t1, qdegree=12, face_order=6,
                 time_order=6, volume_subdivisions=1, trace_rays=False):
        self.space = space
        mesh = space.mesh
        self.flux_kernel = cp.RawKernel(FLUX_KERNEL, "flux_means")
        self.flux_kernel.compile()
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        q = quadrature(max(qdegree, 2*space.degree), volume_subdivisions)
        physical = mesh.points(q).reshape(-1, 2)
        feet = cp.ascontiguousarray(flow(physical, t1, t0))
        self.source, refs = self.locate(feet)
        self.target = cp.repeat(cp.arange(mesh.cells, dtype=cp.int32), len(q))
        self.sb = space.basis(refs)
        self.tb = cp.tile(space.basis(q[:, :2]), (mesh.cells, 1))
        self.weights = (2*mesh.areas[:, None]*q[None, :, 2]).ravel()
        self.count = len(self.source)
        s, ws = gauss_interval(face_order)
        theta, wt = gauss_interval(time_order)
        a = mesh.triangles
        e = cp.roll(a, -1, axis=1)-a
        edges = a[:, :, None, :]+s[None, None, :, None]*e[:, :, None, :]
        # All cells are positively oriented. The same edge has opposite normal
        # in its neighbour; symmetric Gauss nodes yield identical flux samples.
        normal_measure = cp.stack((e[..., 1], -e[..., 0]), axis=-1)
        points = cp.broadcast_to(edges[:, :, :, None, :], (mesh.cells, 3, face_order, time_order, 2))
        feet = cp.empty_like(points)
        velocities = cp.empty_like(points)
        for k in range(time_order):
            current = t0+(t1-t0)*theta[k]
            feet[:, :, :, k] = flow(edges, current, t0)
            velocities[:, :, :, k] = velocity(edges, current)
        self.face_source, face_refs = self.locate(cp.ascontiguousarray(feet.reshape(-1, 2)))
        self.face_basis = space.basis(face_refs)
        self.face_target = cp.repeat(cp.arange(mesh.cells, dtype=cp.int32), 3*face_order*time_order)
        flux = cp.sum(velocities*normal_measure[:, :, None, None], axis=-1)
        self.face_weights = cp.ascontiguousarray(((t1-t0)*flux*ws[None, None, :, None]*wt[None, None, None, :]).ravel())
        self.face_count = len(self.face_source)
        if trace_rays:
            # This option is restricted to autonomous straight characteristics:
            # translation and the tested horizontal shear satisfy this exactly.
            source = f"#define NB {space.nb}\n#define M_SQRT2 1.41421356237309504880\n"
            source += (HERE/"basis.cuh").read_text()+(HERE/"transfer.cu").read_text()+(HERE/"ray_flux.cu").read_text()
            module = cp.RawModule(code=source, options=("--std=c++17",))
            kernel = module.get_function("integrate_rays")
            origins = cp.ascontiguousarray(edges.reshape(-1, 2))
            displacements = cp.ascontiguousarray(-(t1-t0)*velocity(edges, t1).reshape(-1, 2))
            # The 3x3 locator and slot bound inherit the jittered-grid scope.
            # Overflow is detected and never silently truncated.
            slots = int(cp.ceil(cp.max(cp.abs(displacements))*mesh.n*12))+16
            owners = cp.empty((len(origins), slots), dtype=cp.int32)
            moments = cp.empty((len(origins), slots, space.nb), dtype=cp.float64)
            nodes, weights = gauss_interval(max(1, (space.degree+2)//2))
            errors = cp.zeros(3, dtype=cp.int32)
            launch(kernel, len(origins), (mesh.triangles, mesh.n, origins, displacements,
                   len(origins), slots, nodes, weights, len(nodes), owners, moments, errors))
            if bool(cp.any(errors)):
                raise RuntimeError(f"Ray tracing failed [location, progress, capacity]: {errors.get()}")
            active = cp.flatnonzero(owners.ravel() >= 0)
            ray = active//slots
            self.face_source = owners.ravel()[active]
            self.face_basis = cp.ascontiguousarray(moments.reshape(-1, space.nb)[active])
            self.face_target = (ray//(3*face_order)).astype(cp.int32)
            flux = cp.sum(velocity(edges, t1)*normal_measure[:, :, None], axis=-1)
            measure = ((t1-t0)*flux*ws[None, None, :]).ravel()
            self.face_weights = cp.ascontiguousarray(measure[ray])
            self.face_count = len(active)
        cp.cuda.Stream.null.synchronize()
        self.build_seconds = time.perf_counter()-start

    def locate(self, points):
        mesh = self.space.mesh
        cells = cp.empty(len(points), dtype=cp.int32)
        refs = cp.empty_like(points)
        errors = cp.zeros(1, dtype=cp.int32)
        launch(self.space.module.get_function("locate_points"), len(points),
               (mesh.triangles, mesh.n, points, len(points), cells, refs, errors))
        if int(errors[0]):
            raise RuntimeError("Characteristic point location failed")
        return cells, refs

    def apply(self, coefficients, output=None):
        if output is None:
            output = cp.zeros_like(coefficients)
        else:
            if output.data.ptr == coefficients.data.ptr:
                raise ValueError("In-place transfer is not defined")
            output.fill(0)
        fields = coefficients.shape[1]
        launch(self.space.module.get_function("weighted_particles"), self.count*fields,
               (self.source, self.target, self.sb, self.tb, self.weights, self.space.mesh.areas,
                self.count, coefficients, output, fields))
        output[:, :, 0] = coefficients[:, :, 0]
        launch(self.flux_kernel, self.face_count*fields,
               (self.face_source, self.face_target, self.face_basis, self.face_weights,
                self.space.mesh.areas, self.face_count, coefficients, output, fields, self.space.nb))
        return output


class StableFluxTransfer(FluxTransfer):
    """Apply a scalar convex energy constraint after the characteristic update.

    The low state is the cellwise P0 projection of the input. Both low and high
    states conserve mass; their convex combination preserves shared face-flux
    conservation. The largest alpha in [0,1] satisfying the input L2 bound is
    computed from a quadratic equation. Accuracy and activation are measured;
    this energy bound alone does not prove convergence or perturbation stability.
    """
    def apply(self, coefficients, output=None):
        high = super().apply(coefficients, output)
        area = self.space.mesh.areas[:, None]
        # Center only the energy calculation to avoid cancellation against a
        # large constant background. The output formula remains conservative.
        mean = self.space.mass(coefficients)/cp.sum(self.space.mesh.areas)
        low_mean = coefficients[:, :, 0]-mean[None, :]
        delta_mean = high[:, :, 0]-coefficients[:, :, 0]
        old_high_energy = cp.sum(area[:, :, None]*coefficients[:, :, 1:]**2, axis=(0, 2))
        new_high_energy = cp.sum(area[:, :, None]*high[:, :, 1:]**2, axis=(0, 2))
        a = cp.sum(area*delta_mean**2, axis=0)+new_high_energy
        b = 2*cp.sum(area*low_mean*delta_mean, axis=0)
        c = -old_high_energy
        discriminant = cp.maximum(b*b-4*a*c, 0)
        root = cp.sqrt(discriminant)
        tiny = cp.finfo(cp.float64).tiny
        alpha = cp.where(b >= 0, -2*c/cp.maximum(b+root, tiny),
                         (-b+root)/cp.maximum(2*a, tiny))
        alpha = cp.where(a+b+c <= 0, 1., cp.clip(alpha, 0., 1.))
        high[:, :, 0] = coefficients[:, :, 0]+alpha[None, :]*delta_mean
        high[:, :, 1:] *= alpha[None, :, None]
        self.last_alpha = alpha
        return high
