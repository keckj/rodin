"""Exercise cuFFT graph ownership across changing large grid sizes.

Distributed under the Boost Software License, Version 1.0.
The 383,383,511,511 sequence reproduced stale-plan reuse before isolation.
"""

import argparse
import gc
import json
from pathlib import Path

from gpu import cp
from gpu_graph import EvolutionGraph
from fourier_modes import FourierModeFlow, GENERAL
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    records = []
    sizes = (383, 383, 511, 511) if args.small else (383, 383, 511, 511, 767, 767) * 2
    previous = None
    for n in sizes:
        flow = FourierModeFlow(n, GENERAL, (0.07, -0.03))
        initial = cp.random.RandomState(2).normal(size=(n, n, 1))
        graph = EvolutionGraph(flow.step, initial, 0.01)
        cp.fft.config.get_plan_cache().clear()
        pressure = cp.zeros((1024, 1024))
        graph.launch(0.01)
        actual = graph.result().copy()
        ordinary = flow.step(initial, 0.01)
        difference = float(cp.max(cp.abs(ordinary - actual)))
        assert difference < 3e-12, difference
        if previous is not None:
            old, old_flow, old_initial = previous
            old.reset(old_initial)
            old.launch(0.01)
            second = float(
                cp.max(cp.abs(old.result() - old_flow.step(old_initial, 0.01)))
            )
            assert second < 3e-12, second
        else:
            second = None
        records.append(
            dict(
                n=n,
                difference=difference,
                previous_graph_difference=second,
                pressure_bytes=pressure.nbytes,
            )
        )
        previous = graph, flow, initial
        del actual, ordinary, pressure
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(status="ok", records=records, provenance=provenance()), indent=2
        )
        + "\n"
    )
    print(json.dumps(dict(status="ok", checks=len(records))))


if __name__ == "__main__":
    main()
