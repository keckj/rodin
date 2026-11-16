"""Incremental FE transport through a static Cartesian DG auxiliary space.

Distributed under the Boost Software License, Version 1.0.
T = I + R* (S-I) R, with an exact L2 projection R and its mass adjoint.
All numerical construction and applications execute on CUDA.
"""
import time

from gpu import cp, HERE, quadrature, launch
from flux import gauss_interval


class Carrier:
    def __init__(self, space, n=None, degree=None, qdegree=None):
        self.space = space
        self.n = n or space.mesh.n
        self.np = (space.degree if degree is None else degree)+1
        self.nc = self.np**2
        self.nodes, self.weights = gauss_interval(self.np)
        self.mass = cp.ascontiguousarray((self.weights[:, None]*self.weights[None, :]).ravel()/self.n**2)
        code = f"#define NB {space.nb}\n#define NP {self.np}\n#define NC {self.nc}\n#define M_SQRT2 1.41421356237309504880\n"
        code += (HERE/"basis.cuh").read_text()+(HERE/"transfer.cu").read_text()+(HERE/"carrier.cu").read_text()
        self.module = cp.RawModule(code=code, options=("--std=c++17",))
        self.builder = self.module.get_function("carrier_blocks")
        self.projector = self.module.get_function("project_carrier")
        self.returner = self.module.get_function("adjoint_carrier")
        self.shifter = self.module.get_function("shift_carrier")
        q = quadrature(qdegree or max(1, space.degree+2*(self.np-1)))
        bounds = cp.floor(space.mesh.triangles*self.n)
        widths = cp.max(bounds, axis=1)-cp.min(bounds, axis=1)+3
        slots = int(cp.max(widths[:, 0]*widths[:, 1]))
        sources, targets, matrices = [], [], []
        errors = cp.zeros(1, dtype=cp.int32)
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        for first in range(0, space.mesh.cells, 2048):
            count = min(2048, space.mesh.cells-first)
            destination = cp.empty(count*slots, dtype=cp.int32)
            blocks = cp.empty((count*slots, self.nc, space.nb), dtype=cp.float64)
            launch(self.builder, count*slots, (space.mesh.triangles, self.n, first, count,
                   slots, q, len(q), self.nodes, self.weights, destination, blocks, errors))
            active = cp.flatnonzero(destination >= 0)
            sources.append((first+active//slots).astype(cp.int32))
            targets.append(destination[active])
            matrices.append(blocks[active])
        if int(errors[0]):
            raise RuntimeError("Static projection candidate capacity exceeded")
        self.source = cp.concatenate(sources)
        self.target = cp.concatenate(targets)
        self.blocks = cp.concatenate(matrices)
        self.pairs = len(self.source)
        cp.cuda.Stream.null.synchronize()
        self.build_seconds = time.perf_counter()-start

    def project(self, coefficients):
        fields = coefficients.shape[1]
        output = cp.zeros((self.n*self.n, fields, self.nc), dtype=cp.float64)
        launch(self.projector, self.pairs*fields*self.nc,
               (self.source, self.target, self.blocks, self.pairs, coefficients, output, fields))
        return output

    def adjoint(self, coefficients):
        fields = coefficients.shape[1]
        output = cp.zeros((self.space.mesh.cells, fields, self.space.nb), dtype=cp.float64)
        launch(self.returner, self.pairs*fields*self.space.nb,
               (self.source, self.target, self.blocks, self.space.mesh.areas, self.mass,
                self.pairs, coefficients, output, fields))
        return output

    def lagrange(self, x):
        values = cp.ones((*x.shape, self.np), dtype=cp.float64)
        for i in range(self.np):
            for j in range(self.np):
                if i != j:
                    values[..., i] *= (x-self.nodes[j])/(self.nodes[i]-self.nodes[j])
        return values

    def transport(self, dt, flow="translation"):
        y = (cp.arange(self.n)[:, None]+self.nodes[None, :]).ravel()/self.n
        if flow == "translation":
            velocities = [cp.ones_like(y), cp.full_like(y, .37)]
        elif flow == "shear":
            velocities = [.2*cp.sin(2*cp.pi*y)]
        else:
            raise ValueError("Carrier prototype supports translation and horizontal shear")
        operators = []
        for velocity in velocities:
            displacement = self.n*dt*velocity
            shift = cp.floor(displacement).astype(cp.int32)
            fraction = displacement-shift
            left = fraction[:, None]*self.nodes[None, :]
            right = fraction[:, None]+(1-fraction[:, None])*self.nodes[None, :]
            points = cp.stack((left, right), axis=1)
            origins = points-fraction[:, None, None]+cp.asarray([1., 0.])[None, :, None]
            lengths = cp.stack((fraction, 1-fraction), axis=1)
            test = self.lagrange(points)
            trial = self.lagrange(origins)
            blocks = cp.einsum("rsqi,rsqj,q,rs->rsij", test, trial, self.weights, lengths)
            blocks /= self.weights[None, None, :, None]
            operators.append((shift, cp.ascontiguousarray(blocks)))
        return IncrementalTransfer(self, operators)


class IncrementalTransfer:
    def __init__(self, carrier, operators):
        self.carrier = carrier
        self.operators = operators

    def apply(self, coefficients, incremental=True):
        c = self.carrier
        old = c.project(coefficients)
        new = old
        fields = coefficients.shape[1]
        for axis, (shifts, blocks) in enumerate(self.operators):
            output = cp.empty_like(new)
            launch(c.shifter, new.size, (new, output, c.n, fields, shifts, blocks, axis))
            new = output
        if incremental:
            return coefficients+c.adjoint(new-old)
        return c.adjoint(new)
