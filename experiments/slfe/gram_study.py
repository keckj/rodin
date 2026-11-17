"""Independent GPU eigensystem checks and interleaved tiny-Gram timings.

Distributed under the Boost Software License, Version 1.0.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

from gpu import cp
from small_gram import eigh
from campaign import provenance


def check(n, fields, spectrum):
    rng = cp.random.RandomState(417 + n + fields)
    orthogonal, _ = cp.linalg.qr(rng.normal(size=(fields, n, n)))
    eigen = cp.linspace(0, 1, n) if spectrum == "singular" else cp.logspace(-14, 0, n)
    matrix = (orthogonal * eigen[None, None, :]) @ orthogonal.transpose(0, 2, 1)
    matrix = (matrix + matrix.transpose(0, 2, 1)) / 2
    expected, _ = cp.linalg.eigh(matrix)
    actual, vectors = eigh(matrix)
    difference = float(cp.max(cp.abs(actual - expected)))
    residual = float(cp.max(cp.abs(matrix @ vectors - vectors * actual[:, None, :])))
    orthogonality = float(
        cp.max(cp.abs(vectors.transpose(0, 2, 1) @ vectors - cp.eye(n)))
    )
    assert max(difference, residual, orthogonality) < 2e-12, (
        difference,
        residual,
        orthogonality,
    )
    timings = {"cupy": [], "jacobi": []}
    solvers = {"cupy": cp.linalg.eigh, "jacobi": eigh}
    for repeat in range(5):
        for name in ("cupy", "jacobi") if repeat % 2 == 0 else ("jacobi", "cupy"):
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            solvers[name](matrix)
            cp.cuda.Stream.null.synchronize()
            timings[name].append(time.perf_counter() - start)
    return dict(
        order=n,
        fields=fields,
        spectrum=spectrum,
        eigenvalue_error=difference,
        residual=residual,
        orthogonality_error=orthogonality,
        seconds=timings,
        median_seconds={k: statistics.median(v) for k, v in timings.items()},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    with args.output.open("w") as stream:
        for n in (1, 2, 3, 5, 12, 20):
            for spectrum in ("singular", "small"):
                result = check(n, 4 if n <= 5 else 1, spectrum)
                stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
                stream.flush()
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
