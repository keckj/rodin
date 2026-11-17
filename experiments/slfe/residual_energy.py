"""Positive CUDA integration of a physical FE--auxiliary residual.

Distributed under the Boost Software License, Version 1.0.
This diagnostic resolves cancellation before squaring, at common quadrature
points. It does not subtract two nearly equal integrated energies. Degree
2*max(p,2*q) integrates the square of P_p minus Q_q on affine intersections.
"""

from gpu import cp, HERE, quadrature, launch


class ResidualEnergy:
    def __init__(self, coupling, subdivisions=1):
        self.coupling = coupling
        c = coupling
        self.quadrature = quadrature(
            2 * max(c.space.degree, 2 * (c.np - 1)), subdivisions
        )
        code = f"#define NB {c.space.nb}\n#define NP {c.np}\n#define NC {c.nc}\n#define NONUNIFORM {int(c.nonuniform)}\n#define M_SQRT2 1.41421356237309504880\n"
        for name in (
            "basis.cuh",
            "transfer.cu",
            "tensor_intervals.cuh",
            "carrier.cu",
            "residual_energy.cu",
        ):
            code += (HERE / name).read_text()
        self.module = cp.RawModule(code=code, options=("--std=c++17",))
        self.kernel = self.module.get_function("residual_energy")

    def apply(self, fe, auxiliary):
        c = self.coupling
        if fe.shape != (
            c.space.mesh.cells,
            auxiliary.shape[1],
            c.space.nb,
        ) or auxiliary.shape != (c.n**2, fe.shape[1], c.nc):
            raise ValueError("Expected matching FE and auxiliary field arrays")
        if fe.dtype != cp.float64 or auxiliary.dtype != cp.float64:
            raise ValueError("Positive residual quadrature uses double precision")
        fields = fe.shape[1]
        output = cp.empty((c.space.mesh.cells * c.candidate_slots, fields))
        errors = cp.zeros(1, dtype=cp.int32)
        launch(
            self.kernel,
            output.size,
            (
                c.space.mesh.triangles,
                c.n,
                c.edges,
                c.space.mesh.cells,
                c.candidate_slots,
                self.quadrature,
                len(self.quadrature),
                c.nodes,
                cp.ascontiguousarray(fe),
                cp.ascontiguousarray(auxiliary),
                fields,
                output,
                errors,
            ),
        )
        if int(errors[0]):
            raise RuntimeError("Residual quadrature candidate capacity exceeded")
        return cp.sum(output, axis=0)
