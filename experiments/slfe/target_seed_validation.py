"""Independent GPU controls for retaining the raw auxiliary target moment.

Distributed under the Boost Software License, Version 1.0.
Containing P_W f guarantees an error no larger than its best W approximation.
Dense joint mass matrices and positive residual quadrature check the claim.
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
from reaction_moments_validation import joint_matrix, coordinates
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = cp.random.RandomState(929)
    controls = []
    for degree in (1, 2, 4):
        space = Space(Mesh(4, jitter=0.18, seed=816), degree)
        carrier = Carrier(space, n=4, builder="tiled")
        base = MemoryTransport(
            carrier, rng.normal(size=(space.mesh.cells, 2, space.nb))
        )
        detail = rng.normal(size=base.auxiliary.shape)
        base.auxiliary = carrier.project(base.anchor) + detail
        base.value = base.anchor + carrier.adjoint(detail)
        gram = joint_matrix(carrier)
        linear = ReactionMoments(
            carrier, 0.2, diagnostics=True, subdivisions=2, builder="tiled"
        )
        nonlinear = QuadratureMoments(carrier, 0.3, reaction="cubic", subdivisions=2)
        for name, operator in (("linear", linear), ("cubic", nonlinear)):
            if name == "linear":
                observation = operator.apply_fe(base.anchor) + operator.cross[
                    1
                ].adjoint(detail)
                moment = operator.cross[1].project(
                    base.anchor
                ) + operator.apply_auxiliary(detail)
                coefficient = coordinates(base)
                squared = joint_matrix(carrier, operator, 2)
                target_energy = cp.sum(coefficient * (squared @ coefficient), axis=0)
            else:
                observation, moment, _, target_energy, _ = operator.moments(
                    base.anchor, detail
                )
            best_w_energy = carrier.energy(moment)
            for shared in (False, True):
                state = copy(base)
                operator.apply(state, iterations=1, shared=shared, target_seed=True)
                coefficient = coordinates(state)
                actual_energy = cp.sum(coefficient * (gram @ coefficient), axis=0)
                scale = cp.max(target_energy)
                containment = float(cp.max((best_w_energy - actual_energy) / scale))
                contraction = float(cp.max((actual_energy - target_energy) / scale))
                observable = float(cp.max(cp.abs(state.value - observation)))
                assert max(containment, contraction, observable) < 1e-10, (
                    degree,
                    name,
                    shared,
                    containment,
                    contraction,
                    observable,
                )
                controls.append(
                    dict(
                        degree=degree,
                        reaction=name,
                        shared=shared,
                        normalized_best_w_excess=containment,
                        normalized_contraction_excess=contraction,
                        observation_error=observable,
                    )
                )
        # A target in W is retained exactly even when its incoming direction
        # is unrelated. Test the physical error through positive quadrature.
        target = rng.normal(size=base.auxiliary.shape)
        incoming = rng.normal(size=target.shape)
        state = MemoryTransport(carrier, cp.zeros_like(base.value))
        observation = carrier.adjoint(target)
        state.project_moments(
            carrier,
            observation,
            target - carrier.project(observation),
            incoming,
            iterations=1,
            shared=True,
            search_seed=target,
            gram_backend="reduction",
        )
        d = state.auxiliary - carrier.project(state.anchor)
        error = ResidualEnergy(carrier).apply(state.anchor, target - d)
        relative = float(cp.sqrt(cp.max(error / carrier.energy(target))))
        assert relative < 1e-10, relative
        controls.append(
            dict(degree=degree, exact_auxiliary_target_relative_error=relative)
        )
    args.output.write_text(
        json.dumps(dict(provenance=provenance(), controls=controls), indent=2) + "\n"
    )
    print(json.dumps(dict(controls=len(controls), passed=True)))


if __name__ == "__main__":
    main()
