"""Independent GPU ambient oracles for coupled physical reaction moments.

Distributed under the Boost Software License, Version 1.0.
Checks component covariance, mixture closure, energy, and physical observation.
"""

import argparse
from copy import copy
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from carrier import Carrier
from memory_transport import MemoryTransport
from spectral_reaction_moments import SpectralReactionMoments
from reaction_moments_validation import coordinates, joint_matrix
from system_coupling_study import mix
from campaign import provenance


def cycle_matrix():
    return cp.asarray(
        [
            [2.0, -1.0, 0.0, -1.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [-1.0, 0.0, -1.0, 2.0],
        ]
    )


def validate(degree):
    space = Space(Mesh(4, jitter=0.18, seed=817), degree)
    coupling = Carrier(space, n=4)
    random = cp.random.RandomState(813)
    original = MemoryTransport(coupling, cp.zeros((space.mesh.cells, 4, space.nb)))
    original.anchor = random.normal(size=original.anchor.shape)
    original.anchor[:, -1] = -cp.sum(original.anchor[:, :-1], axis=1)
    original.anchor[:, -1, 0] += 1
    detail = random.normal(size=original.auxiliary.shape)
    detail[:, -1] = -cp.sum(detail[:, :-1], axis=1)
    original.auxiliary = coupling.project(original.anchor) + detail
    original.value = original.anchor + coupling.adjoint(detail)
    gram = joint_matrix(coupling)
    rotation, _ = cp.linalg.qr(random.normal(size=(4, 4)))
    matrix = cycle_matrix()
    records = []
    for duration in (0.0, 0.1, 0.5):
        operator = SpectralReactionMoments(coupling, matrix, duration, diagnostics=True)
        rotated_operator = SpectralReactionMoments(
            coupling, rotation.T @ matrix @ rotation, duration
        )
        coefficients = coordinates(original) @ operator.rotation
        target_moments = cp.empty_like(coefficients)
        target_energy = cp.asarray(0.0)
        for field, scalar in enumerate(operator.operators):
            weighted = gram if scalar is None else joint_matrix(coupling, scalar)
            squared = gram if scalar is None else joint_matrix(coupling, scalar, 2)
            target_moments[:, field] = weighted @ coefficients[:, field]
            target_energy += coefficients[:, field] @ squared @ coefficients[:, field]
        for shared in (False, True):
            actual = copy(original)
            result = operator.apply(actual, shared=shared)
            new = coordinates(actual) @ operator.rotation
            energy = cp.sum(new * (gram @ new))
            distance = target_energy + energy - 2 * cp.sum(new * target_moments)
            pythagoras_error = float(cp.abs(distance - (target_energy - energy)))
            target_energy_error = float(
                cp.abs(result["total_target_energy"] - target_energy)
            )
            normalized_observation = (
                (
                    mix(actual.value, operator.rotation)
                    * cp.sqrt(space.mesh.areas)[:, None, None]
                )
                .transpose(0, 2, 1)
                .reshape(-1, 4)
            )
            observation_error = float(
                cp.max(
                    cp.abs(
                        normalized_observation
                        - target_moments[: len(normalized_observation)]
                    )
                )
            )
            rotated = copy(original)
            for attribute in ("value", "anchor", "auxiliary"):
                setattr(rotated, attribute, mix(getattr(original, attribute), rotation))
            rotated_operator.apply(rotated, shared=shared)
            covariance = max(
                float(
                    cp.max(
                        cp.abs(
                            getattr(actual, key)
                            - mix(getattr(rotated, key), rotation.T)
                        )
                    )
                )
                for key in ("value", "auxiliary")
            )
            constant = cp.zeros((space.mesh.cells, 1, space.nb))
            constant[:, :, 0] = 1
            closure = float(
                cp.sqrt(
                    space.energy(
                        cp.sum(actual.value, axis=1, keepdims=True) - constant
                    )[0]
                )
            )
            auxiliary_closure = float(
                cp.sqrt(
                    coupling.energy(
                        cp.sum(actual.auxiliary, axis=1, keepdims=True) - 1
                    )[0]
                )
            )
            excess = float(result["total_coercivity_excess"])
            assert (
                max(pythagoras_error, target_energy_error, observation_error, excess)
                < 2e-9
            )
            if shared:
                assert max(covariance, closure, auxiliary_closure) < 2e-8, (
                    covariance,
                    closure,
                    auxiliary_closure,
                )
            uniform_error = None
            if duration == 0:
                uniform_error = max(
                    float(cp.max(cp.abs(getattr(actual, key) - getattr(original, key))))
                    for key in ("value", "auxiliary")
                )
                assert uniform_error < 1e-8, uniform_error
            records.append(
                dict(
                    degree=degree,
                    duration=duration,
                    shared=shared,
                    pythagoras_error=pythagoras_error,
                    target_energy_error=target_energy_error,
                    observation_error=observation_error,
                    covariance_error=covariance,
                    fe_closure_l2=closure,
                    auxiliary_closure_l2=auxiliary_closure,
                    coercivity_excess=excess,
                    identity_error=uniform_error,
                )
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    records = []
    for degree in (1, 2, 4):
        records.extend(validate(degree))
        args.output.write_text(
            json.dumps(dict(results=records, provenance=meta), indent=2) + "\n"
        )
        print(json.dumps(dict(degree=degree, controls=len(records))), flush=True)


if __name__ == "__main__":
    main()
