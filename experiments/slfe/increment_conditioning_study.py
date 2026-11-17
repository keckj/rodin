"""Track retained old memory and coefficient conditioning during stiff decay.

Distributed under the Boost Software License, Version 1.0.
These diagnostic repetitions are excluded from timing benchmarks.
"""

import argparse
import json
from pathlib import Path

from gpu import cp, Space
from graded_study import GradedMesh
from moving_study import profile_family
from carrier import Carrier
from tensor_cellular import TensorCellular
from memory_transport import MemoryTransport
from reaction_moments import ReactionMoments
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive", action="store_true")
    args = parser.parse_args()
    meta = provenance()
    space = Space(GradedMesh(128, 0.95), 4)
    initial = space.project(profile_family(0.01, 1), degree=20)
    carrier = Carrier(space, n=256, builder="tiled")
    if args.positive:
        from residual_energy import ResidualEnergy

        positive = ResidualEnergy(carrier)
    flow = TensorCellular(carrier, 0.25 / 256, grading=0.0)
    with args.output.open("w") as stream:
        for reanchor in (False, True):
            operator = ReactionMoments(
                carrier,
                100 * 0.25 / (2 * 256),
                increment=True,
                reanchor=reanchor,
                builder="tiled",
            )
            state = MemoryTransport(carrier, initial)
            history = []
            for step in range(256):
                for half in (0, 1):
                    result = operator.apply(state)
                    d = state.auxiliary - carrier.project(state.anchor)
                    visible = state.anchor + carrier.adjoint(d)
                    history.append(
                        dict(
                            step=step,
                            half=half,
                            time=(step + (half + 1) / 2) * 0.25 / 256,
                            anchor_energy=float(space.energy(state.anchor)[0]),
                            detail_energy=float(carrier.energy(d)[0]),
                            visible_energy=float(space.energy(state.value)[0]),
                            ambient_energy=float(state.energy()[0]),
                            incoming_dropped_energy=float(
                                result["incoming_projection_loss"][0]
                            ),
                            rank=int(result["retained_rank"]),
                            observation_defect=float(
                                cp.sqrt(space.energy(visible - state.value)[0])
                            ),
                        )
                    )
                    if args.positive and (step % 16 == 0 or step == 255):
                        history[-1]["positive_ambient_energy"] = float(
                            positive.apply(state.anchor, -d)[0]
                        )
                    if half == 0:
                        state.advect(flow.apply_carrier)
            record = dict(
                nfe=128,
                naux=256,
                strength=100,
                steps=256,
                reanchor=reanchor,
                positive_quadrature=args.positive,
                history=history,
            )
            stream.write(json.dumps(dict(result=record, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps(
                    dict(
                        reanchor=reanchor,
                        final=history[-1],
                        maximum_dropped_energy=max(
                            x["incoming_dropped_energy"] for x in history
                        ),
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
