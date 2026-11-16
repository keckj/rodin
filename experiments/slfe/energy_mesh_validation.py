"""CUDA controls for energy-based meshes and the coupled multicomponent path.

Distributed under the Boost Software License, Version 1.0.
The independent reference refinement checks characteristic and reaction errors.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, launch, Space
from graded_study import GradedMesh
from energy_mesh import partitions
from moving_study import profile_family
from reaction_study import reference_kernel
from tensor_cellular import TensorCellular
from carrier import Carrier
from adaptive_study import evolve
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coordinate = (cp.arange(512) + 0.5) / 512
    yy, xx = cp.meshgrid(coordinate, coordinate, indexing="ij")
    points = cp.stack((xx, yy), axis=-1)
    uniform, oriented = partitions(points, cp.ones(points.shape[:-1]), 32)
    uniform_error = float(
        cp.max(
            cp.abs(cp.concatenate((uniform, oriented)) - cp.linspace(0, 1, 33)[None, :])
        )
    )
    assert uniform_error < 2e-11, uniform_error
    profile = profile_family(0.01, 4, spread=0.15)
    base, oriented = partitions(points, cp.sum(profile(points, 0) ** 2, axis=-1), 32)
    space = Space(GradedMesh(8, 0.5), 4)
    carrier = Carrier(space, n=32, edges=base)
    flow = TensorCellular(carrier, 0.07, oriented_edges=oriented)
    one = cp.ones((carrier.n**2, 1, carrier.nc))
    constant_error = float(cp.max(cp.abs(flow.apply_carrier(one) - 1)))
    random = cp.random.RandomState(921).normal(size=(carrier.n**2, 8, carrier.nc))
    result = flow.apply_carrier(random)
    energy_ratio = float(cp.max(carrier.energy(result) / carrier.energy(random)))
    mass_error = float(
        cp.max(
            cp.abs(cp.sum(carrier.mass[:, None, :] * (result - random), axis=(0, 2)))
        )
    )
    assert constant_error < 3e-10 and mass_error < 3e-10 and energy_ratio < 1 + 3e-10
    test_points = cp.random.RandomState(998).uniform(0, 1, (16384, 2))
    references = []
    for count in (512, 1024):
        traced = cp.empty((len(test_points), 3))
        launch(
            reference_kernel(),
            len(test_points),
            (test_points, traced, len(test_points), 0.25, count),
        )
        references.append(profile(traced[:, :2], 0) * cp.exp(traced[:, 2, None]))
    reference_error = float(cp.max(cp.abs(references[0] - references[1])))
    assert reference_error < 1e-7, reference_error
    regressions = []
    for reaction, expected in (
        (False, 0.0037390215109742364),
        (True, 0.002728018954599365),
    ):
        measured = evolve(
            nfe=64,
            naux=64,
            end=0.25,
            steps=128,
            reaction=reaction,
            projection_iterations=8,
        )
        error = abs(measured["l2"][0] - expected)
        assert error < 2e-11, error
        regressions.append(dict(reaction=reaction, discrepancy=error))
    record = dict(
        uniform_error=uniform_error,
        constant_error=constant_error,
        mass_error=mass_error,
        random_energy_ratio=energy_ratio,
        reference_512_1024_error=reference_error,
        scalar_regressions=regressions,
        provenance=provenance(),
    )
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
