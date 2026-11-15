"""Check the proposed marginal-cubature theorem on the actual CUDA operator.

Distributed under the Boost Software License, Version 1.0.
"""
import argparse
import json
from pathlib import Path

from gpu import cp, Mesh, Space, translation, constant, smooth
from moments import MomentTransfer
from flows import compressible, compressible_jacobian, compressible_solution
from research import moment_case
from cartesian import CartesianTransfer, coordinates


def run():
    records = []
    for p in [1, 2]:
        result = moment_case(4, p, 3, courant=.731)
        assert result["accepted_moments"], result
        assert result["accepted_positive"], result
        assert result["constant_error"] < 1e-10, result
        assert result["random_mass_error"] < 1e-10, result
        assert result["singular_max"] <= 1+1e-10, result
        records.append(result)
    isolated = moment_case(4, 1, 1, fit="signed", courant=1e-5, steps=1)
    assert isolated["crossing_fraction"] == 0, isolated
    assert isolated["isolated_cells"] == 32, isolated
    assert isolated["relative_moment_residual"] > 1e-6, isolated
    records.append({"check": "small_translation_isolated_cell_obstruction", "result": isolated})
    mesh = Mesh(4)
    space = Space(mesh, 1)
    # The original 3-way subdivision failed this positive feasibility search.
    # Retain it as a reported counterexample, then enrich the support to six.
    insufficient = MomentTransfer(space, compressible, 0, .1, qdegree=12,
                                  subdivisions=3, fit="positive", solver="dense",
                                  jacobian=compressible_jacobian)
    assert insufficient.relative_moment_residual > 1e-6
    records.append({"check": "insufficient_positive_support_reported",
                    "summary": insufficient.summary(),
                    "interpretation": "Restricted active-set failure, not an infeasibility proof"})
    transfer = MomentTransfer(space, compressible, 0, .1, qdegree=12,
                              subdivisions=6, fit="positive", solver="dense",
                              jacobian=compressible_jacobian)
    assert transfer.relative_moment_residual < 1e-10, transfer.summary()
    assert transfer.minimum_weight_ratio >= -1e-12, transfer.summary()
    random = cp.random.RandomState(931).normal(size=(mesh.cells, 1, space.nb))
    y = transfer.apply(random)
    mass_error = float(cp.max(cp.abs(space.mass(y)-space.mass(random))))
    ratio = float(cp.max(space.energy(y)/space.energy(random)))
    assert mass_error < 1e-10, mass_error
    assert ratio <= 1/transfer.jacobian_min+1e-10, ratio
    # Invalid Jacobian cannot silently invoke a volume-preserving shortcut.
    try:
        MomentTransfer(space, compressible, 0, .1,
                       jacobian=lambda x, t0, t1: -cp.ones(x.shape[0]))
    except ValueError:
        pass
    else:
        raise AssertionError("Negative flow Jacobian accepted")
    # Mutation establishes sensitivity of mass and constant gates.
    one = cp.zeros_like(random)
    one[:, :, 0] = 1.
    original_mass = space.mass(transfer.apply(one))
    transfer.weights *= .99
    assert float(cp.max(cp.abs(space.mass(transfer.apply(one))-original_mass))) > .009
    records.append({"check": "compressible_mass_and_bound", "mass_error": mass_error,
                    "energy_ratio": ratio, "bound": 1/transfer.jacobian_min})
    for kernel in ["L2_1", "L4_2"]:
        t = CartesianTransfer(64, .037, kernel)
        for axis in range(2):
            z = t.shifts[axis].astype(cp.float64)
            moments = 2 if kernel == "L2_1" else 4
            displacement = (1., .37)[axis]*.037*64
            for degree in range(moments+1):
                observed = float(cp.sum(t.weights[axis]*(z-displacement)**degree))
                assert abs(observed-int(degree == 0)) < 1e-11, (kernel, axis, degree, observed)
        q = smooth(coordinates(64), 0)[..., None]
        mass_error = abs(float(cp.mean(t.apply(q))-cp.mean(q)))
        assert mass_error < 1e-12, mass_error
        records.append({"check": "lambda_gpu_moments", "kernel": kernel, "mass_error": mass_error})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = run()
    args.output.write_text(json.dumps({"passed": True, "records": records}, indent=2)+"\n")
    print(f"Passed {len(records)} moment/cubature validation groups")
