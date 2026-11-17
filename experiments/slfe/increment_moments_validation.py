"""GPU validation of physical-increment projection down to tiny time steps.

Distributed under the Boost Software License, Version 1.0.
Dense joint Grams independently evaluate increment energies and residuals.
The small squared defect is evaluated at its own scale, without subtracting
full state energies. Zero-duration identity is checked coefficient by coefficient.
"""

import argparse
from copy import copy
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from carrier import Carrier
from memory_transport import MemoryTransport
from reaction_moments import ReactionMoments
from reaction_moments_validation import joint_matrix, coordinates
from campaign import provenance


def validate(degree):
    space = Space(Mesh(4, jitter=0.18, seed=817), degree)
    c = Carrier(space, n=4)
    rng = cp.random.RandomState(141)
    original = MemoryTransport(c, cp.zeros((space.mesh.cells, 2, space.nb)))
    original.anchor = rng.normal(size=original.anchor.shape)
    detail = rng.normal(size=original.auxiliary.shape)
    original.auxiliary = c.project(original.anchor) + detail
    original.value = original.anchor + c.adjoint(detail)
    coefficient = coordinates(original)
    gram = joint_matrix(c)
    records = []
    for duration in (0.0, 1e-12, 1e-8, 1e-4, 0.01, 0.2):
        for amplitude in (0.0, 0.5):
            operator = ReactionMoments(
                c, duration, amplitude, increment=True, diagnostics=True
            )
            first = joint_matrix(c, operator)
            second = joint_matrix(c, operator, 2)
            increment_energy = cp.sum(coefficient * (second @ coefficient), axis=0)
            actual = copy(original)
            result = operator.apply(actual)
            projected = copy(original)
            projected.value, projected.anchor, projected.auxiliary = result[
                "increment_coefficients"
            ]
            delta = coordinates(projected)
            delta_energy = cp.sum(delta * (gram @ delta), axis=0)
            inner = cp.sum(coefficient * (first @ delta), axis=0)
            distance = increment_energy + delta_energy - 2 * inner
            scale = cp.maximum(increment_energy, 1e-300)
            defect_error = float(
                cp.max(cp.abs(distance - result["energy_loss"]) / scale)
            )
            increment_energy_error = float(
                cp.max(cp.abs(increment_energy - result["increment_energy"]) / scale)
            )
            observation = (
                c.adjoint(actual.auxiliary)
                + actual.anchor
                - c.adjoint(c.project(actual.anchor))
            )
            observation_error = float(cp.max(cp.abs(actual.value - observation)))
            assert max(defect_error, increment_energy_error) < 2e-8, (
                defect_error,
                increment_energy_error,
            )
            assert (
                observation_error < 2e-11
                and float(cp.max(result["coercivity_excess"])) < 2e-10
            )
            identity = None
            if duration == 0:
                identity = all(
                    bool(cp.array_equal(getattr(actual, key), getattr(original, key)))
                    for key in ("value", "anchor", "auxiliary")
                )
                assert identity, degree
            uniform_matrix_error = None
            if amplitude == 0 and duration > 0:
                weight = cp.expm1(cp.asarray(-duration))
                uniform_matrix_error = float(cp.max(cp.abs(first / weight - gram)))
                assert uniform_matrix_error < 2e-11, uniform_matrix_error
            records.append(
                dict(
                    degree=degree,
                    duration=duration,
                    amplitude=amplitude,
                    defect_relative_oracle_error=defect_error,
                    increment_energy_relative_error=increment_energy_error,
                    observation_error=observation_error,
                    bitwise_zero_step_identity=identity,
                    uniform_scaled_matrix_error=uniform_matrix_error,
                    increment_energy=increment_energy.tolist(),
                    projection_loss=result["energy_loss"].tolist(),
                    projection_loss_per_duration_squared=(
                        (result["energy_loss"] / duration**2).tolist()
                        if duration
                        else None
                    ),
                )
            )
        print(json.dumps(dict(degree=degree, duration=duration)), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    results = []
    for degree in (1, 2, 4):
        results.extend(validate(degree))
        args.output.write_text(
            json.dumps(dict(results=results, provenance=meta), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
