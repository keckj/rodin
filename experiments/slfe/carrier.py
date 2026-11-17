"""Incremental FE transport through a static Cartesian DG auxiliary space.

Distributed under the Boost Software License, Version 1.0.
T = I + R* (S-I) R, with an exact L2 projection R and its mass adjoint.
All numerical construction and applications execute on CUDA.
"""

import time
import json
import math

from gpu import cp, HERE, quadrature, launch
from flux import gauss_interval


class Carrier:
    def __init__(
        self,
        space,
        n=None,
        degree=None,
        qdegree=None,
        builder="auto",
        edges=None,
        reaction_weight=None,
        subdivisions=1,
        reaction_form="value",
    ):
        self.space = space
        self.reaction_weight = reaction_weight
        self.reaction_form = reaction_form
        if reaction_form not in ("value", "increment", "increment-square"):
            raise ValueError("Unknown reaction weight form")
        self.n = n or space.mesh.n
        self.nonuniform = edges is not None
        self.edges = cp.ascontiguousarray(
            (
                cp.stack((cp.linspace(0.0, 1.0, self.n + 1),) * 2)
                if edges is None
                else edges
            ),
            dtype=cp.float64,
        )
        if self.edges.shape != (2, self.n + 1):
            raise ValueError("Expected two periodic partitions with n + 1 edges")
        if not bool(cp.all(cp.diff(self.edges, axis=1) > 0)) or not bool(
            cp.all(cp.abs(self.edges[:, -1] - self.edges[:, 0] - 1) < 1e-13)
        ):
            raise ValueError("Partitions must increase strictly and have period one")
        self.np = (space.degree if degree is None else degree) + 1
        if not 1 <= self.np <= 6:
            raise ValueError("Carrier polynomial degrees 0 through 5 are available")
        self.nc = self.np**2
        self.nodes, self.weights = gauss_interval(self.np)
        self.mass = cp.ascontiguousarray(
            (self.weights[:, None] * self.weights[None, :]).ravel() / self.n**2
        )
        if self.nonuniform:
            sizes = cp.diff(self.edges, axis=1)
            areas = (sizes[1, :, None] * sizes[0, None, :]).ravel()
            self.mass = cp.ascontiguousarray(
                areas[:, None] * (self.weights[:, None] * self.weights[None, :]).ravel()
            )
        code = f"#define NB {space.nb}\n#define NP {self.np}\n#define NC {self.nc}\n#define M_SQRT2 1.41421356237309504880\n"
        code += f"#define NONUNIFORM {int(self.nonuniform)}\n"
        if reaction_weight is not None:
            duration, amplitude = map(float, reaction_weight)
            if (
                not math.isfinite(duration)
                or not duration >= 0
                or not -1 <= amplitude <= 1
            ):
                raise ValueError(
                    "Expected nonnegative reaction duration and amplitude in [-1,1]"
                )
            # Weighted cross mass for reaction experiments. Such an object
            # supplies P_W g|_V and its mass adjoint, not an unweighted
            # projection coupling for MemoryTransport.
            exponent = f"-({duration:.17g})*(1+({amplitude:.17g})*sin(6.28318530717958647693*(position).x))"
            expression = (
                f"exp({exponent})" if reaction_form == "value" else f"expm1({exponent})"
            )
            if reaction_form == "increment-square":
                expression = f"({expression})*({expression})"
            code += f"#define CARRIER_WEIGHT(position) ({expression})\n"
        code += (
            (HERE / "basis.cuh").read_text()
            + (HERE / "transfer.cu").read_text()
            + (HERE / "tensor_intervals.cuh").read_text()
            + (HERE / "carrier.cu").read_text()
            + (HERE / "tensor_remap.cu").read_text()
        )
        if builder == "tiled":
            code += (HERE / "carrier_tiled.cu").read_text()
        self.module = cp.RawModule(code=code, options=("--std=c++17",))
        # Cooperative assembly pays off for the high-order blocks measured in
        # carrier_build_study.py. Keep the small scalar blocks on their kernel.
        if builder == "auto":
            builder = "warp" if space.nb * self.nc >= 32 else "scalar"
        if builder not in ("scalar", "warp", "tiled"):
            raise ValueError("Expected scalar, warp or tiled assembly")
        self.builder = self.module.get_function(
            "carrier_blocks" if builder == "scalar" else "carrier_blocks_" + builder
        )
        self.projector = self.module.get_function("project_carrier")
        self.returner = self.module.get_function("adjoint_carrier")
        self.shifter = self.module.get_function("shift_carrier")
        self.shear_builder = self.module.get_function("build_shear_blocks")
        table = json.loads((HERE / "carrier_polynomials.json").read_text())["tables"][
            str(self.np - 1)
        ]
        self.bernstein_degree = table["bernstein_degree"]
        modal = cp.asarray(table["modal_coefficients"], dtype=cp.float64)
        legendre = [cp.ones_like(self.nodes)]
        z = 2 * self.nodes - 1
        for k in range(1, self.np):
            legendre.append(
                z
                if k == 1
                else ((2 * k - 1) * z * legendre[k - 1] - (k - 1) * legendre[k - 2]) / k
            )
        modal_values = (
            cp.stack(legendre, axis=1) * cp.sqrt(2 * cp.arange(self.np) + 1)[None, :]
        )
        self.polynomials = cp.ascontiguousarray(
            cp.einsum(
                "ik,skld,jl,j->sijd", modal_values, modal, modal_values, self.weights
            )
        )
        # Exact endpoint matrices avoid adding projection roundoff at zero motion.
        self.polynomials[0, :, :, 0] = 0
        self.polynomials[1, :, :, 0] = cp.eye(self.np)
        self.polynomials[0, :, :, -1] = cp.eye(self.np)
        self.polynomials[1, :, :, -1] = 0
        q = quadrature(
            qdegree or max(1, space.degree + 2 * (self.np - 1)), subdivisions
        )
        bounds = cp.floor(space.mesh.triangles * self.n)
        if self.nonuniform:
            for axis in (0, 1):
                values = space.mesh.triangles[..., axis]
                periods = cp.floor(values - self.edges[axis, 0])
                bounds[..., axis] = (
                    cp.searchsorted(self.edges[axis], values - periods, side="right")
                    - 1
                    + self.n * periods
                )
        widths = cp.max(bounds, axis=1) - cp.min(bounds, axis=1) + 3
        slots = int(cp.max(widths[:, 0] * widths[:, 1]))
        self.candidate_slots = slots
        sources, targets, matrices = [], [], []
        errors = cp.zeros(1, dtype=cp.int32)
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        for first in range(0, space.mesh.cells, 2048):
            count = min(2048, space.mesh.cells - first)
            destination = cp.empty(count * slots, dtype=cp.int32)
            blocks = cp.empty((count * slots, self.nc, space.nb), dtype=cp.float64)
            arguments = (
                space.mesh.triangles,
                self.n,
                self.edges,
                first,
                count,
                slots,
                q,
                len(q),
                self.nodes,
                self.weights,
                destination,
                blocks,
                errors,
            )
            if builder == "scalar":
                launch(self.builder, count * slots, arguments)
            else:
                self.builder(((count * slots + 3) // 4,), (128,), arguments)
            active = cp.flatnonzero(destination >= 0)
            sources.append((first + active // slots).astype(cp.int32))
            targets.append(destination[active])
            matrices.append(blocks[active])
        if int(errors[0]):
            raise RuntimeError("Static projection candidate capacity exceeded")
        self.source = cp.concatenate(sources)
        self.target = cp.concatenate(targets)
        self.blocks = cp.concatenate(matrices)
        self.pairs = len(self.source)
        cp.cuda.Stream.null.synchronize()
        self.build_seconds = time.perf_counter() - start

    def energy(self, coefficients):
        """Exact tensor-polynomial L2 norm squared, one entry per field."""
        weights = self.mass[:, None, :] if self.nonuniform else self.mass
        return cp.sum(weights * coefficients**2, axis=(0, 2))

    def project(self, coefficients):
        fields = coefficients.shape[1]
        output = cp.zeros((self.n * self.n, fields, self.nc), dtype=cp.float64)
        launch(
            self.projector,
            self.pairs * fields * self.nc,
            (
                self.source,
                self.target,
                self.blocks,
                self.pairs,
                coefficients,
                output,
                fields,
            ),
        )
        return output

    def adjoint(self, coefficients):
        fields = coefficients.shape[1]
        output = cp.zeros(
            (self.space.mesh.cells, fields, self.space.nb), dtype=cp.float64
        )
        launch(
            self.returner,
            self.pairs * fields * self.space.nb,
            (
                self.source,
                self.target,
                self.blocks,
                self.space.mesh.areas,
                self.mass,
                self.pairs,
                coefficients,
                output,
                fields,
            ),
        )
        return output

    def transport(self, dt, flow="translation"):
        if self.nonuniform:
            coordinates = [
                (
                    self.edges[axis, :-1, None]
                    + cp.diff(self.edges[axis])[:, None] * self.nodes
                ).ravel()
                for axis in (0, 1)
            ]
            if flow == "translation":
                velocities = [
                    cp.ones_like(coordinates[1]),
                    cp.full_like(coordinates[0], 0.37),
                ]
            elif flow == "shear":
                velocities = [0.2 * cp.sin(2 * cp.pi * coordinates[1])]
            else:
                raise ValueError("Expected translation or shear")
            return self.shears([(axis, dt, v) for axis, v in enumerate(velocities)])
        y = (cp.arange(self.n)[:, None] + self.nodes[None, :]).ravel() / self.n
        if flow == "translation":
            velocities = [cp.ones_like(y), cp.full_like(y, 0.37)]
        elif flow == "shear":
            velocities = [0.2 * cp.sin(2 * cp.pi * y)]
        else:
            raise ValueError(
                "Carrier prototype supports translation and horizontal shear"
            )
        return self.shears(
            [(axis, dt, velocity) for axis, velocity in enumerate(velocities)]
        )

    def shears(self, stages):
        """Compose volume-preserving Cartesian shears before returning an increment.

        Each stage is (axis, duration, velocity_at_transverse_Gauss_nodes).
        Axis 0 advects x with a velocity depending only on y; axis 1 is the
        corresponding y shear. The input velocity array has n*(q+1) entries.
        """
        if self.nonuniform:
            from tensor_remap import TensorTransfer

            return TensorTransfer(self, stages)
        operators = []
        for axis, dt, velocity in stages:
            if axis not in (0, 1) or velocity.shape != (self.n * self.np,):
                raise ValueError(
                    "Expected an axis and one velocity per transverse Gauss node"
                )
            shift = cp.empty(velocity.size, dtype=cp.int32)
            blocks = cp.empty((velocity.size, 2, self.np, self.np), dtype=cp.float64)
            duration = cp.asarray(dt, dtype=cp.float64)
            launch(
                self.shear_builder,
                blocks.size,
                (
                    velocity,
                    duration,
                    self.n,
                    velocity.size,
                    self.bernstein_degree,
                    self.polynomials,
                    shift,
                    blocks,
                ),
            )
            operators.append((axis, shift, blocks))
        return IncrementalTransfer(self, operators)


class IncrementalTransfer:
    def __init__(self, carrier, operators):
        self.carrier = carrier
        self.operators = operators

    def apply_carrier(self, coefficients):
        c = self.carrier
        new = coefficients
        fields = coefficients.shape[1]
        for axis, shifts, blocks in self.operators:
            output = cp.empty_like(new)
            launch(
                c.shifter, new.size, (new, output, c.n, fields, shifts, blocks, axis)
            )
            new = output
        return new

    def apply(self, coefficients, incremental=True):
        c = self.carrier
        old = c.project(coefficients)
        new = self.apply_carrier(old)
        if incremental:
            return coefficients + c.adjoint(new - old)
        return c.adjoint(new)
