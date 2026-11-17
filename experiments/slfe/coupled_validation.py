"""Focused GPU controls for the new FE physics and remapping controllers.

Distributed under the Boost Software License, Version 1.0.
Designed for Compute Sanitizer. Its runtimes are not performance measurements.
"""

import argparse
import json
from pathlib import Path

from diffusion_study import validate as diffusion_validate, evolve as diffusion_evolve
from warm_validation import validate as memory_validate
from phase_study import validate as phase_validate
from budgeted_roundtrip import validate as budget_validate
from source_study import evolve as source_evolve
from roundtrip_auxiliary import evolve as auxiliary_evolve
from physics_fields import validate_sampler
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(
        diffusion=diffusion_validate(),
        coupled_heat=diffusion_evolve(nfe=8, steps=4),
        memory=memory_validate(2, methods=("galerkin0", "galerkin2")),
        minimax=phase_validate("maximum-displacement"),
        budget=budget_validate(),
        source=source_evolve(
            nfe=16, naux=8, width=0.03, steps=8, source_tolerance=0.02, maximum=16
        ),
        auxiliary=auxiliary_evolve(nfe=16, naux=8, steps=8, half_duration=0.25),
        sampler=validate_sampler(),
        provenance=provenance(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
