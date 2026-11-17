"""Independent GPU controls for fused pointwise reaction moments.

Distributed under the Boost Software License, Version 1.0.
Linear reaction is compared to separately assembled weighted operators.
Cubic reaction uses constant solutions, positive energy and quadrature
refinement; the final projection is checked through its physical norm.
"""

import argparse
from copy import copy
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from carrier import Carrier
from memory_transport import MemoryTransport
from reaction_moments import ReactionMoments
from quadrature_moments import QuadratureMoments
from residual_energy import ResidualEnergy
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    records = []
    random = cp.random.RandomState(713)
    for degree in (1, 2, 4):
        space = Space(Mesh(4, jitter=0.15, seed=718), degree)
        for shifted in (False, True):
            line = cp.linspace(0, 1, 5)
            edges = (
                cp.stack(
                    (
                        line + 0.07 + 0.02 * cp.sin(2 * cp.pi * line),
                        line - 0.04 - 0.03 * cp.sin(2 * cp.pi * line),
                    )
                )
                if shifted
                else None
            )
            c = Carrier(space, n=4, edges=edges, builder="tiled")
            initial = MemoryTransport(c, cp.zeros((space.mesh.cells, 2, space.nb)))
            initial.anchor = 0.05 * random.normal(size=initial.anchor.shape)
            detail = 0.05 * random.normal(size=initial.auxiliary.shape)
            initial.auxiliary = c.project(initial.anchor) + detail
            initial.value = initial.anchor + c.adjoint(detail)
            positive = ResidualEnergy(c)
            for reaction, duration in (("linear", 0.2), ("cubic", 0.0), ("cubic", 0.2)):
                operator = QuadratureMoments(c, duration, reaction=reaction)
                v, w, old_energy, target_energy, _ = operator.moments(
                    initial.anchor, detail
                )
                old_error = float(
                    cp.max(cp.abs(old_energy - positive.apply(initial.anchor, -detail)))
                )
                assert old_error < 1e-12
                weighted_error = None
                refined_weighted_error = None
                if reaction == "linear":
                    reference = ReactionMoments(
                        c, duration, diagnostics=True, builder="tiled", subdivisions=2
                    )
                    expected_v = reference.apply_fe(initial.anchor) + reference.cross[
                        1
                    ].adjoint(detail)
                    expected_w = reference.cross[1].project(
                        initial.anchor
                    ) + reference.apply_auxiliary(detail)
                    weighted_error = max(
                        float(cp.max(cp.abs(v - expected_v))),
                        float(cp.max(cp.abs(w - expected_w))),
                    )
                elif duration == 0:
                    weighted_error = max(
                        float(cp.max(cp.abs(v - initial.value))),
                        float(cp.max(cp.abs(w - initial.auxiliary))),
                    )
                    assert weighted_error < 1e-11
                refined = QuadratureMoments(
                    c, duration, reaction=reaction, subdivisions=2
                )
                rv, rw, _, re, _ = refined.moments(initial.anchor, detail)
                if reaction == "linear":
                    refined_weighted_error = max(
                        float(cp.max(cp.abs(rv - expected_v))),
                        float(cp.max(cp.abs(rw - expected_w))),
                    )
                    assert refined_weighted_error < 1e-11, refined_weighted_error
                refined_error = max(
                    float(cp.max(cp.abs(v - rv))),
                    float(cp.max(cp.abs(w - rw))),
                    float(cp.max(cp.abs(target_energy - re))),
                )
                assert refined_error < 1e-3, refined_error
                second_refinement = None
                if reaction == "cubic" and duration:
                    fine = QuadratureMoments(c, duration, subdivisions=4)
                    fv, fw, _, fe, _ = fine.moments(initial.anchor, detail)
                    second_refinement = max(
                        float(cp.max(cp.abs(rv - fv))),
                        float(cp.max(cp.abs(rw - fw))),
                        float(cp.max(cp.abs(re - fe))),
                    )
                    assert second_refinement < max(0.25 * refined_error, 1e-12)
                state = copy(initial)
                operator.apply(state)
                new_detail = state.auxiliary - c.project(state.anchor)
                after = positive.apply(state.anchor, -new_detail)
                mass = cp.broadcast_to(c.mass, (c.n**2, c.nc))
                inner = cp.sum(
                    state.anchor * v * space.mesh.areas[:, None, None], axis=(0, 2)
                ) + cp.sum(new_detail * w * mass[:, None, :], axis=(0, 2))
                pythagoras = float(cp.max(cp.abs(2 * (after - inner))))
                assert (
                    pythagoras < 1e-10 and float(cp.max(after - target_energy)) < 1e-10
                )
                assert float(cp.max(target_energy - old_energy)) < 1e-12
                reference = 0.7 * initial.anchor
                _, _, _, _, target_distance = operator.moments(
                    initial.anchor, detail, reference=reference
                )
                _, _, _, _, local_defect = operator.moments(
                    initial.anchor, cp.zeros_like(detail), reference=reference
                )
                old_distance = positive.apply(cp.zeros_like(reference), -detail)
                new_distance = positive.apply(state.anchor - reference, -new_detail)
                relative_identity = float(
                    cp.max(
                        cp.abs(new_distance - target_distance + target_energy - after)
                    )
                )
                relative_bound_excess = float(
                    cp.max(
                        cp.sqrt(new_distance)
                        - cp.sqrt(old_distance)
                        - cp.sqrt(local_defect)
                    )
                )
                assert relative_identity < 1e-11
                assert relative_bound_excess < 1e-11
                record = dict(
                    degree=degree,
                    shifted=shifted,
                    reaction=reaction,
                    duration=duration,
                    input_energy_error=old_error,
                    linear_oracle_error=weighted_error,
                    refined_linear_oracle_error=refined_weighted_error,
                    quadrature_refinement_error=refined_error,
                    second_refinement_error=second_refinement,
                    pythagoras_error=pythagoras,
                    relative_identity_error=relative_identity,
                    relative_bound_excess=relative_bound_excess,
                    energy_ratio=float(cp.max(after / old_energy)),
                )
                records.append(record)
                print(json.dumps(record), flush=True)
            constant = cp.zeros_like(initial.anchor)
            constant[:, :, 0] = cp.asarray((0.3, 2.0))[None, :]
            state = MemoryTransport(c, constant)
            operator = QuadratureMoments(c, 0.4, amplitude=0.0)
            operator.apply(state)
            expected = constant / cp.sqrt(
                1 + 0.8 * cp.asarray((0.3, 2.0))[None, :, None] ** 2
            )
            constant_error = float(cp.max(cp.abs(state.value - expected)))
            assert constant_error < 1e-11, constant_error
            records.append(
                dict(
                    degree=degree,
                    shifted=shifted,
                    reaction="constant cubic",
                    constant_error=constant_error,
                )
            )
            args.output.write_text(
                json.dumps(dict(results=records, provenance=meta), indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
