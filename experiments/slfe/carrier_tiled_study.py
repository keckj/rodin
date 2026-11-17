"""Validate parallel quadrature staging against the scalar/warp CUDA builders.

Distributed under the Boost Software License, Version 1.0.
Timings are interleaved after warming each builder. Numerical blocks and
actual forward/adjoint applications are compared before reporting costs.
"""

import argparse
import json
from pathlib import Path
import time
from statistics import median

from gpu import cp, Mesh, Space
from graded_study import GradedMesh
from carrier import Carrier
from memory_transport import MemoryTransport
from tensor_cellular import TensorCellular
from moving_study import profile_family
from energy_mesh import from_fe
from campaign import provenance


def compare(a, b):
    assert bool(cp.all(a.source == b.source)) and bool(cp.all(a.target == b.target))
    block_error = float(cp.max(cp.abs(a.blocks - b.blocks)))
    relative_block_error = block_error / max(float(cp.max(cp.abs(a.blocks))), 1)
    random = cp.random.RandomState(383)
    v = random.normal(size=(a.space.mesh.cells, 4, a.space.nb))
    w = random.normal(size=(a.n**2, 4, a.nc))
    pv, pa = a.project(v), a.adjoint(w)
    forward = float(cp.max(cp.abs(pv - b.project(v))))
    backward = float(cp.max(cp.abs(pa - b.adjoint(w))))
    mass = cp.broadcast_to(a.mass, (a.n**2, a.nc))
    adjoint_error = float(
        cp.abs(
            cp.einsum("cfi,ci,cfi->", b.project(v), mass, w)
            - cp.einsum("cfi,c,cfi->", v, a.space.mesh.areas, b.adjoint(w))
        )
    )
    assert (
        relative_block_error < 5e-12
        and max(forward, backward) < 2e-9
        and adjoint_error < 1e-11
    ), (block_error, forward, backward, adjoint_error)
    constant_error = None
    if a.reaction_weight is None:
        one = cp.zeros((a.space.mesh.cells, 1, a.space.nb))
        one[:, :, 0] = 1
        constant_error = max(
            float(cp.max(cp.abs(b.project(one) - 1))),
            float(cp.max(cp.abs(b.adjoint(cp.ones((b.n**2, 1, b.nc))) - one))),
        )
        assert constant_error < 5e-9, constant_error
    return dict(
        block_error=block_error,
        relative_block_error=relative_block_error,
        forward_error=forward,
        adjoint_application_error=backward,
        mass_adjoint_error=adjoint_error,
        constant_error=constant_error,
    )


def validate():
    records = []
    for degree in (1, 2, 4):
        space = Space(Mesh(8, jitter=0.18, seed=817), degree)
        x = cp.linspace(0, 1, 9)
        edges = cp.stack(
            (
                x + 0.1 * cp.sin(2 * cp.pi * x) + 0.13,
                x - 0.08 * cp.sin(2 * cp.pi * x) - 0.17,
            )
        )
        for nonuniform in (False, True):
            for weighted in (False, True):
                for subdivisions in (1, 2):
                    options = dict(
                        n=8,
                        edges=edges if nonuniform else None,
                        reaction_weight=(0.2, 0.5) if weighted else None,
                        qdegree=20 if weighted else None,
                        subdivisions=subdivisions,
                    )
                    scalar = Carrier(space, builder="scalar", **options)
                    tiled = Carrier(space, builder="tiled", **options)
                    records.append(
                        dict(
                            degree=degree,
                            nonuniform=nonuniform,
                            weighted=weighted,
                            subdivisions=subdivisions,
                            **compare(scalar, tiled)
                        )
                    )
        print(json.dumps(dict(validated_degree=degree)), flush=True)
    return records


def benchmark(nfe, repeats):
    space = Space(GradedMesh(nfe, 0.95), 4)
    initial = space.project(profile_family(0.01, 4, spread=0.15), degree=20)
    edges, oriented = from_fe(space, initial, nfe // 2)
    options = dict(n=nfe // 2, edges=edges)
    warp = Carrier(space, builder="warp", **options)
    tiled = Carrier(space, builder="tiled", **options)
    comparison = compare(warp, tiled)
    sa, sb = MemoryTransport(warp, initial), MemoryTransport(tiled, initial)
    fa = TensorCellular(warp, 0.25 / 128, oriented_edges=oriented)
    fb = TensorCellular(tiled, 0.25 / 128, oriented_edges=oriented)
    for _ in range(16):
        sa.advect(fa.apply_carrier)
        sb.advect(fb.apply_carrier)
    transport_l2 = float(cp.sqrt(cp.sum(space.energy(sa.value - sb.value))))
    assert transport_l2 < 1e-10, transport_l2
    samples = {builder: [] for builder in ("warp", "tiled")}
    for repeat in range(repeats):
        for builder in ("warp", "tiled") if repeat % 2 == 0 else ("tiled", "warp"):
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            carrier = Carrier(space, builder=builder, **options)
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - start
            samples[builder].append(
                dict(total_seconds=elapsed, build_seconds=carrier.build_seconds)
            )
            del carrier
    return dict(
        nfe=nfe,
        naux=nfe // 2,
        samples=samples,
        median_total_seconds={
            builder: median(x["total_seconds"] for x in values)
            for builder, values in samples.items()
        },
        transport_l2=transport_l2,
        **comparison
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    result = dict(provenance=provenance(), controls=validate())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not args.validation_only:
        result["timings"] = []
        for nfe in (128, 256):
            result["timings"].append(benchmark(nfe, 5))
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result["timings"][-1]), flush=True)
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
