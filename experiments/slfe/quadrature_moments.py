"""Pointwise physical reaction followed by GPU shared-memory compression.

Distributed under the Boost Software License, Version 1.0.
Positive common quadrature measures the complete field before applying a
linear or cubic reaction. Both FE and auxiliary moments come from the same
samples. No frozen, field-dependent reaction matrix is substituted.
"""

import math

from gpu import cp, HERE, quadrature


class QuadratureMoments:
    def __init__(
        self,
        coupling,
        duration,
        reaction="cubic",
        amplitude=0.5,
        qdegree=20,
        subdivisions=1,
    ):
        if reaction not in ("linear", "cubic"):
            raise ValueError("Expected linear or cubic pointwise reaction")
        if not math.isfinite(duration) or duration < 0 or not -1 <= amplitude <= 1:
            raise ValueError(
                "Expected finite nonnegative duration and amplitude in [-1,1]"
            )
        self.coupling, self.duration, self.amplitude = coupling, duration, amplitude
        self.reaction = 0 if reaction == "linear" else 1
        c = coupling
        self.quadrature = quadrature(
            max(qdegree, 2 * max(c.space.degree, 2 * (c.np - 1))), subdivisions
        )
        code = f"#define NB {c.space.nb}\n#define NP {c.np}\n#define NC {c.nc}\n#define NONUNIFORM {int(c.nonuniform)}\n#define M_SQRT2 1.41421356237309504880\n"
        for name in (
            "basis.cuh",
            "transfer.cu",
            "tensor_intervals.cuh",
            "carrier.cu",
            "quadrature_moments.cu",
        ):
            code += (HERE / name).read_text()
        self.module = cp.RawModule(code=code, options=("--std=c++17",))
        self.kernel = self.module.get_function("quadrature_moments")

    def moments(self, anchor, detail, reference=None):
        """Return FE/W moments, input/target energies and optional FE defect.

        The fifth result is the positive quadrature of |F(q)-reference|^2,
        or None when no finite element reference is supplied.
        """
        c = self.coupling
        if anchor.ndim != 3 or detail.ndim != 3:
            raise ValueError("Expected three-dimensional coefficient arrays")
        if anchor.dtype != cp.float64 or detail.dtype != cp.float64:
            raise ValueError("Physical moment quadrature uses double precision")
        if reference is not None and (
            reference.shape != anchor.shape or reference.dtype != cp.float64
        ):
            raise ValueError("The reference must match the double FE coefficients")
        if anchor.shape != (
            c.space.mesh.cells,
            detail.shape[1],
            c.space.nb,
        ) or detail.shape != (c.n**2, anchor.shape[1], c.nc):
            raise ValueError("Expected compatible FE and auxiliary components")
        fields = anchor.shape[1]
        observation, moment = cp.zeros_like(anchor), cp.zeros_like(detail)
        count = c.space.mesh.cells * c.candidate_slots * fields
        energies = cp.zeros((c.space.mesh.cells * c.candidate_slots, fields, 3))
        errors = cp.zeros(1, dtype=cp.int32)
        self.kernel(
            ((count + 3) // 4,),
            (128,),
            (
                c.space.mesh.triangles,
                c.n,
                c.edges,
                c.space.mesh.cells,
                c.candidate_slots,
                self.quadrature,
                len(self.quadrature),
                c.nodes,
                c.weights,
                c.space.mesh.areas,
                cp.ascontiguousarray(anchor),
                cp.ascontiguousarray(detail),
                cp.ascontiguousarray(anchor if reference is None else reference),
                int(reference is not None),
                fields,
                float(self.duration),
                float(self.amplitude),
                self.reaction,
                observation,
                moment,
                energies,
                errors,
            ),
        )
        if int(errors[0]):
            raise RuntimeError("Physical moments candidate capacity exceeded")
        totals = cp.sum(energies, axis=0)
        return (
            observation,
            moment,
            totals[:, 0],
            totals[:, 1],
            None if reference is None else totals[:, 2],
        )

    def apply(self, state, iterations=2, shared=True):
        c = self.coupling
        if state.coupling is not c:
            raise ValueError("The common quadrature belongs to a fixed support")
        detail = state.auxiliary - c.project(state.anchor)
        observation, moment, before, target, _ = self.moments(state.anchor, detail)
        result = state.project_moments(
            c,
            observation,
            moment - c.project(observation),
            detail,
            target_energy=target,
            iterations=iterations,
            shared=shared,
            gram_backend="reduction",
        )
        result.update(positive_input_energy=before, positive_target_energy=target)
        return result
