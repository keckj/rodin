"""Focused memory-safety and independent graph checks for the day prototypes.

Distributed under the Boost Software License, Version 1.0.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Mesh, Space, gaussian
from gpu_graph import EvolutionGraph
from fourier_carrier import UnitaryCellular
from fourier_flow import NestedCellularTransport
from spectral_reference import SpectralTransport
from day_study import kernel_validation
import reaction_study
import fourier_study
from campaign import provenance


def graph_validation():
    checks = []
    for n in (31, 63, 127):
        for method in ("unitary", "rk4"):
            flow = (
                UnitaryCellular(n) if method == "unitary" else SpectralTransport(n + 1)
            )
            shape = (n, n, 1) if method == "unitary" else (n + 1, n + 1)
            initial = cp.random.RandomState(41).normal(size=shape)
            ordinary = initial
            graph = EvolutionGraph(flow.step, initial, 0.001)
            cp.fft.config.get_plan_cache().clear()
            # Exercise captured-plan lifetime and allocation isolation.
            pressure = cp.zeros((1024, 1024))
            for dt in (0.001, 0.002, -0.001):
                ordinary = flow.step(ordinary, dt)
                graph.launch(dt)
            error = float(cp.max(cp.abs(ordinary - graph.result())))
            assert error < 2e-12, error
            checks.append(
                dict(n=n, method=method, error=error, pressure_bytes=pressure.nbytes)
            )
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    space = Space(Mesh(4), 2)
    coefficients = space.project(gaussian)
    transformed = NestedCellularTransport(space).step(0, 0.1).apply(coefficients)
    assert (
        float(space.energy(transformed)[0])
        <= float(space.energy(coefficients)[0]) + 1e-12
    )
    result = dict(
        status="ok",
        provenance=provenance(),
        kernels=kernel_validation(),
        graphs=graph_validation(),
        reaction=reaction_study.validate(),
        coupled_case=reaction_study.evolve(n=8, p=2, steps=2, modes=31),
        fourier_case=fourier_study.evolve(n=8, p=2, steps=2, modes=31, graph=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="ok", graph_checks=len(result["graphs"]))))


if __name__ == "__main__":
    main()
