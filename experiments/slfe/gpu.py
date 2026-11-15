"""CUDA numerical experiment. CuPy is mandatory; no CPU solver fallback.

Distributed under the Boost Software License, Version 1.0.
"""
from functools import lru_cache
from math import factorial
from pathlib import Path
import re
import time

import cupy as cp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


@lru_cache(None)
def module(degree):
    if degree not in range(5):
        raise ValueError("Only basis degrees 0 through 4 are implemented")
    nb = (degree+1)*(degree+2)//2
    source = f"#define NB {nb}\n#define M_SQRT2 1.41421356237309504880\n" + (HERE / "basis.cuh").read_text()
    source += (HERE / "transfer.cu").read_text()
    result = cp.RawModule(code=source, options=("--std=c++17",))
    # Trigger compilation before timing a numerical operation.
    result.get_function("evaluate_basis")
    return result


def launch(kernel, count, args):
    kernel(((count+127)//128,), (128,), args)


def quadrature(degree, subdivisions=1):
    """Reuse Rodin's quadrature constants; subdivide the reference triangle on GPU."""
    degree = max(1, degree)
    if not 1 <= degree <= 20 or subdivisions < 1:
        raise ValueError("Quadrature degree must be 1..20, subdivisions >= 1")
    text = (ROOT / "src/Rodin/QF/XiaoGimbutasData.h").read_text()
    section = text.split("// degree "+str(degree)+":", 1)[1].split("}", 1)[0].split("{", 1)[1]
    data = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", section)]
    q = cp.asarray(data, dtype=cp.float64).reshape(-1, 3)
    if subdivisions == 1:
        return q
    # Enumeration is integer setup. The geometric transformations are GPU work.
    cells = []
    for i in range(subdivisions):
        for j in range(subdivisions-i):
            cells.append([i, j, i+1, j, i, j+1])
            if i+j < subdivisions-1:
                cells.append([i+1, j, i+1, j+1, i, j+1])
    t = cp.asarray(cells, dtype=cp.float64).reshape(-1, 3, 2)/subdivisions
    x = t[:, None, 0] + q[None, :, :1]*(t[:, None, 1]-t[:, None, 0])
    x += q[None, :, 1:2]*(t[:, None, 2]-t[:, None, 0])
    w = cp.broadcast_to(q[None, :, 2]/subdivisions**2, x.shape[:2])
    return cp.ascontiguousarray(cp.concatenate((x, w[..., None]), axis=2).reshape(-1, 3))


class Mesh:
    """Periodic jittered triangular lattice, represented by resident vertex arrays.

    Connectivity is not Cartesian FE connectivity: diagonal orientation varies,
    and cells are affine nonuniform triangles. Candidate search nevertheless
    exploits the lattice ancestry, explicitly limiting this research locator.
    """
    def __init__(self, n, jitter=0.18, seed=17):
        if n < 4 or not 0 <= jitter <= 0.24:
            raise ValueError("n >= 4 and jitter in [0, 0.24] are required")
        self.n = n
        self.jitter = jitter
        self.seed = seed
        random = cp.random.RandomState(seed)
        noise = random.uniform(-jitter, jitter, (n, n, 2)).astype(cp.float64)
        j, i = cp.meshgrid(cp.arange(n+1), cp.arange(n+1), indexing="ij")
        vertices = cp.stack((i, j), axis=-1).astype(cp.float64)
        vertices += noise[j % n, i % n]
        vertices /= n
        a, b = vertices[:-1, :-1], vertices[:-1, 1:]
        c, d = vertices[1:, :-1], vertices[1:, 1:]
        flip = random.randint(0, 2, (n, n)).astype(cp.bool_)
        t0 = cp.stack((a, b, cp.where(flip[..., None], c, d)), axis=-2)
        t1 = cp.stack((cp.where(flip[..., None], b, a), d, c), axis=-2)
        self.triangles = cp.ascontiguousarray(cp.stack((t0, t1), axis=-3).reshape(-1, 3, 2))
        e = self.triangles[:, 1]-self.triangles[:, 0]
        f = self.triangles[:, 2]-self.triangles[:, 0]
        self.areas = (e[:, 0]*f[:, 1]-e[:, 1]*f[:, 0])/2
        if float(cp.min(self.areas)) <= 0:
            raise ValueError("Inverted triangle")
        self.cells = 2*n*n

    def mapped(self, flow, t0, t1):
        return cp.ascontiguousarray(flow(self.triangles, t0, t1))

    def points(self, q):
        t = self.triangles
        return t[:, None, 0]+q[None, :, :1]*(t[:, None, 1]-t[:, None, 0])+q[None, :, 1:2]*(t[:, None, 2]-t[:, None, 0])


class Space:
    def __init__(self, mesh, degree):
        self.mesh = mesh
        self.degree = degree
        self.nb = (degree+1)*(degree+2)//2
        self.module = module(degree)

    def basis(self, points):
        points = cp.ascontiguousarray(points.reshape(-1, 2))
        values = cp.empty((points.shape[0], self.nb), dtype=cp.float64)
        launch(self.module.get_function("evaluate_basis"), points.shape[0], (points, points.shape[0], values))
        return values

    def project(self, function, time=0., degree=16):
        q = quadrature(max(2*self.degree, degree))
        b = self.basis(q[:, :2])
        values = function(self.mesh.points(q), time)
        if values.ndim == 2:
            values = values[..., None]
        result = cp.einsum("cqf,qb,q->cfb", values, b, 2*q[:, 2])
        return cp.ascontiguousarray(result)

    def mass(self, coefficients):
        return cp.sum(self.mesh.areas[:, None]*coefficients[:, :, 0], axis=0)

    def energy(self, coefficients):
        return cp.sum(self.mesh.areas[:, None, None]*coefficients**2, axis=(0, 2))

    def errors(self, coefficients, function, time, degree=18, subdivisions=1):
        q = quadrature(degree, subdivisions)
        basis = self.basis(q[:, :2])
        # Bound peak diagnostic memory by processing element batches.
        l1 = cp.zeros(coefficients.shape[1], dtype=cp.float64)
        l2 = cp.zeros_like(l1)
        linf = cp.zeros_like(l1)
        minimum = cp.full_like(l1, cp.inf)
        maximum = cp.full_like(l1, -cp.inf)
        for first in range(0, self.mesh.cells, 4096):
            last = min(first+4096, self.mesh.cells)
            t = self.mesh.triangles[first:last]
            points = t[:, None, 0]+q[None, :, :1]*(t[:, None, 1]-t[:, None, 0])+q[None, :, 1:2]*(t[:, None, 2]-t[:, None, 0])
            actual = cp.einsum("cfb,qb->cqf", coefficients[first:last], basis)
            exact = function(points, time)
            if exact.ndim == 2:
                exact = exact[..., None]
            error = cp.abs(actual-exact)
            w = 2*self.mesh.areas[first:last, None]*q[None, :, 2]
            l1 += cp.sum(w[..., None]*error, axis=(0, 1))
            l2 += cp.sum(w[..., None]*error**2, axis=(0, 1))
            linf = cp.maximum(linf, cp.max(error, axis=(0, 1)))
            minimum = cp.minimum(minimum, cp.min(actual, axis=(0, 1)))
            maximum = cp.maximum(maximum, cp.max(actual, axis=(0, 1)))
        return {"l1": l1.get().tolist(), "l2": cp.sqrt(l2).get().tolist(),
                "linf_sampled": linf.get().tolist(), "min_sampled": minimum.get().tolist(),
                "max_sampled": maximum.get().tolist()}


class Transfer:
    """Source-oriented conservative block transfer, built entirely on CUDA.

    For geometric mode, source and target polynomial products are integrated
    exactly for a piecewise affine flow (up to roundoff), using intersection
    polygons and existing degree-2p quadrature. A nonlinear flow represented by
    moved vertices is only piecewise affine: geometric consistency is an
    independent approximation, reported explicitly.
    """
    def __init__(self, space, moved, mode="geometric", qdegree=None, subdivisions=1, slots=96):
        if mode not in ("geometric", "quadrature"):
            raise ValueError(mode)
        self.space = space
        self.mode = mode
        self.moved = moved
        if moved.shape != space.mesh.triangles.shape:
            raise ValueError("Moved triangles must match the source mesh")
        q = quadrature(qdegree or max(1, 2*space.degree), subdivisions)
        self.nq = q.shape[0]
        self.quadrature = q
        self.sources, self.destinations, self.blocks = [], [], []
        errors = cp.zeros(2, dtype=cp.int32)
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        for first in range(0, space.mesh.cells, 4096):
            cells = min(4096, space.mesh.cells-first)
            dst = cp.empty(cells*slots, dtype=cp.int32)
            blocks = cp.empty((cells*slots, space.nb, space.nb), dtype=cp.float64)
            launch(space.module.get_function("build_blocks"), cells*slots,
                   (space.mesh.triangles, moved, space.mesh.n, first, cells, slots,
                    q, q.shape[0], int(mode == "quadrature"), dst, blocks, errors))
            active = cp.flatnonzero(dst >= 0)
            self.sources.append((first+active//slots).astype(cp.int32))
            self.destinations.append(dst[active])
            self.blocks.append(blocks[active])
        self.sources = cp.concatenate(self.sources)
        self.destinations = cp.concatenate(self.destinations)
        self.blocks = cp.concatenate(self.blocks)
        cp.cuda.Stream.null.synchronize()
        self.build_seconds = time.perf_counter()-start
        if bool(cp.any(errors)):
            raise RuntimeError(f"Transfer construction failed [inverted, candidate overflow]: {errors.get()}")
        self.pairs = self.sources.size
        self.storage_bytes = self.sources.nbytes+self.destinations.nbytes+self.blocks.nbytes

    def apply(self, coefficients, output=None, transpose=False):
        if coefficients.dtype != cp.float64 or coefficients.shape[0] != self.space.mesh.cells or coefficients.shape[2] != self.space.nb or not coefficients.flags.c_contiguous:
            raise ValueError("Expected contiguous float64 (cells, fields, basis) coefficients")
        if output is None:
            output = cp.zeros_like(coefficients)
        else:
            if output.data.ptr == coefficients.data.ptr:
                raise ValueError("In-place transfer is not defined")
            output.fill(0)
        fields = coefficients.shape[1]
        name = "apply_blocks_transpose" if transpose else "apply_blocks"
        launch(self.space.module.get_function(name), self.pairs*fields*self.space.nb,
               (self.sources, self.destinations, self.blocks, self.pairs, coefficients, output, fields))
        return output

    def deposit(self, coefficients, output=None):
        if output is None:
            output = cp.zeros_like(coefficients)
        else:
            output.fill(0)
        errors = cp.zeros(1, dtype=cp.int32)
        mesh = self.space.mesh
        launch(self.space.module.get_function("deposit_particles"), mesh.cells*self.nq,
               (mesh.triangles, self.moved, mesh.n, self.quadrature, self.nq,
                coefficients, output, coefficients.shape[1], errors))
        if int(errors[0]):
            raise RuntimeError("Particle location failed")
        return output


def translation(points, t0, t1, velocity=(1., 0.37)):
    return points+cp.asarray(velocity, dtype=cp.float64)*(t1-t0)


def shear(points, t0, t1, amplitude=0.2):
    result = points.copy()
    result[..., 0] += amplitude*cp.sin(2*cp.pi*points[..., 1])*(t1-t0)
    return result


def smooth(points, time, velocity=(1., 0.37)):
    x = points[..., 0]-velocity[0]*time
    y = points[..., 1]-velocity[1]*time
    return 1.+0.35*cp.sin(2*cp.pi*x)*cp.cos(2*cp.pi*y)+0.15*cp.cos(4*cp.pi*x-2*cp.pi*y)


def gaussian(points, time, velocity=(1., 0.37)):
    x = points[..., 0]-velocity[0]*time-0.35
    y = points[..., 1]-velocity[1]*time-0.45
    # Smooth periodic localized bump, exactly periodic without a truncated tail.
    return cp.exp(-12*(cp.sin(cp.pi*x)**2+cp.sin(cp.pi*y)**2))


def constant(points, time=0.):
    return cp.ones(points.shape[:-1], dtype=cp.float64)


def timings(operation, repeats=30):
    """CUDA-event timings after warm-up; include all kernels in the operation."""
    for _ in range(3):
        operation()
    cp.cuda.Stream.null.synchronize()
    results = []
    for _ in range(repeats):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        operation()
        end.record()
        end.synchronize()
        results.append(cp.cuda.get_elapsed_time(start, end))
    return sorted(results)[len(results)//2]
