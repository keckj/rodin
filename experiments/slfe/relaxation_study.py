"""GPU test of coercive uniform reaction acting on visible and hidden states.

Distributed under the Boost Software License, Version 1.0.
Uniform reaction commutes with advection. This isolates a subgrid modelling
choice: damping only the FE observation leaves hidden reaction modes undamped.
"""

import argparse
from copy import copy
import json
from pathlib import Path

from gpu import cp, Mesh, Space
from graded_study import GradedMesh
from carrier import Carrier
from tensor_cellular import TensorCellular
from memory_transport import MemoryTransport
from moving_study import profile_family
from krylov_study import ErrorOracle
from campaign import provenance


def validate():
    records = []
    for degree in (1, 2, 4):
        space = Space(Mesh(8, jitter=0.18, seed=817), degree)
        carrier = Carrier(space, n=8)
        initial = cp.random.RandomState(12).normal(size=(space.mesh.cells, 3, space.nb))
        original = MemoryTransport(carrier, initial)
        original.advect(carrier.transport(0.07, "shear").apply_carrier)
        factors = cp.random.RandomState(19).uniform(
            0.2, 0.9, size=(space.mesh.cells, 1, space.nb)
        )
        for gamma in (0.0, 0.1, 0.9, 1.0):
            state = copy(original)
            new = factors * original.value
            expected = space.energy(new) + gamma**2 * (
                original.energy() - space.energy(original.value)
            )
            state.apply_physics(lambda value: factors * value, hidden_decay=gamma)
            defect = float(cp.max(cp.abs(state.energy() - expected)))
            observation = (
                carrier.adjoint(state.auxiliary)
                + state.anchor
                - carrier.adjoint(carrier.project(state.anchor))
            )
            observation_error = float(cp.max(cp.abs(observation - new)))
            assert defect < 1e-10 and observation_error < 1e-10
            records.append(
                dict(
                    degree=degree,
                    hidden_decay=gamma,
                    energy_identity_error=defect,
                    observation_error=observation_error,
                )
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(dict(results=validate(), provenance=meta), indent=2) + "\n"
    )
    with args.output.open("w") as stream:
        for nfe in (32, 64):
            space = Space(GradedMesh(nfe, 0.95), 4)
            profile = profile_family(0.01, 1)
            initial = space.project(profile, degree=20)
            carrier = Carrier(space, n=2 * nfe)
            energy0 = space.energy(initial)
            oracle = ErrorOracle(space, profile, 0.25)
            floor = oracle.errors(oracle.best)["l2"]
            for steps in (16, 64, 256):
                flow = TensorCellular(carrier, 0.25 / steps, grading=0.0)
                baseline = MemoryTransport(carrier, initial)
                for _ in range(steps):
                    baseline.advect(flow.apply_carrier)
                baseline_error = oracle.errors(baseline.value)["l2"]
                for strength in (1.0, 10.0, 100.0):
                    gamma = cp.exp(cp.asarray(-strength * 0.25 / (2 * steps)))
                    final_factor = cp.exp(cp.asarray(-strength * 0.25))
                    states = {
                        method: MemoryTransport(carrier, initial)
                        for method in ("visible-only", "visible-and-hidden")
                    }
                    for _ in range(steps):
                        for method, state in states.items():
                            decay = None if method == "visible-only" else gamma
                            state.apply_physics(
                                lambda value: gamma * value, hidden_decay=decay
                            )
                            state.advect(flow.apply_carrier)
                            state.apply_physics(
                                lambda value: gamma * value, hidden_decay=decay
                            )
                    for method, state in states.items():
                        normalized = state.value / final_factor
                        commutation_error = float(
                            cp.sqrt(space.energy(normalized - baseline.value)[0])
                        )
                        normalized_energy = float(
                            state.energy()[0] / (energy0[0] * final_factor**2)
                        )
                        raw_energy = float(state.energy()[0] / energy0[0])
                        assert raw_energy <= 1 + 1e-8
                        if method == "visible-and-hidden":
                            assert (
                                normalized_energy <= 1 + 1e-8
                                and commutation_error < 1e-9
                            ), (normalized_energy, commutation_error)
                        errors = oracle.errors(normalized)
                        result = dict(
                            nfe=nfe,
                            naux=2 * nfe,
                            steps=steps,
                            end=0.25,
                            strength=strength,
                            method=method,
                            final_decay=float(final_factor),
                            best_fe_l2=floor,
                            undamped_l2=baseline_error,
                            normalized_commutation_l2=commutation_error,
                            normalized_energy_ratio=normalized_energy,
                            energy_ratio=raw_energy,
                            normalized_errors=errors,
                            physical_l2=[float(final_factor) * x for x in errors["l2"]],
                        )
                        stream.write(
                            json.dumps(dict(result=result, provenance=meta)) + "\n"
                        )
                        stream.flush()
                        print(json.dumps(result), flush=True)
            del oracle, carrier
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
