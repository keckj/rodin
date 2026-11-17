"""Focused CUDA checks for the later transport and monitor experiments.

Distributed under the Boost Software License, Version 1.0.
This entry point also runs under Compute Sanitizer; timings are not benchmarks.
"""

import argparse
import json
from pathlib import Path

from svd_study import validate as validate_svd
from compressible_study import validate as validate_compression
from phase_study import validate as validate_phase
from warm_validation import limiter_counterexample
from roundtrip_study import evolve
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = dict(
        svd=validate_svd(),
        compression=validate_compression(),
        phase=validate_phase(),
        limiter_counterexample=limiter_counterexample(),
        shifted_geometry=evolve(nfe=16, naux=16, steps=8, floating_phase=True),
        provenance=provenance(),
    )
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
