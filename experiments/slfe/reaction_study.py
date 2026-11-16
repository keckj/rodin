"""Physical consistency study for persistent FE transport/reaction coupling.

Distributed under the Boost Software License, Version 1.0.
The FE reaction is the exact exponential of its local Galerkin generator.
"""

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

from gpu import cp, HERE, Mesh, Space, gaussian, quadrature, launch
from fourier_carrier import FourierCarrier, UnitaryCellular
from memory_transport import MemoryTransport
from campaign import provenance


class Reaction:
    def __init__(self, space, degree=18):
        code = f"#define NB {space.nb}\n" + (HERE / "local_operator.cu").read_text()
        self.kernel = cp.RawKernel(code, "apply_local_matrix")
        self.kernel.compile()
        q = quadrature(degree)
        basis = space.basis(q[:, :2])
        points = space.mesh.points(q)
        rate = 1 + 0.5 * cp.sin(2 * cp.pi * points[..., 0])
        self.matrix = cp.einsum("cq,qi,qj,q->cij", rate, basis, basis, 2 * q[:, 2])
        self.values, self.vectors = cp.linalg.eigh(self.matrix)
        assert float(self.values.min()) >= 0.5 - 1e-10
        assert float(self.values.max()) <= 1.5 + 1e-10

    def exponential(self, duration):
        decay = cp.exp(-duration * self.values)
        return cp.ascontiguousarray(
            (self.vectors * decay[:, None, :]) @ self.vectors.transpose(0, 2, 1)
        )

    def apply(self, matrix, value):
        result = cp.empty_like(value)
        launch(
            self.kernel,
            value.size,
            (matrix, value, result, value.shape[0], value.shape[1]),
        )
        return result


@lru_cache(None)
def reference_kernel():
    code = (HERE / "cellular_reference.cu").read_text() + (
        HERE / "reaction_reference.cu"
    ).read_text()
    kernel = cp.RawKernel(code, "reaction_reference")
    kernel.compile()
    return kernel


def exact(points, end, substeps=512):
    shape = points.shape[:-1]
    flattened = cp.ascontiguousarray(points.reshape(-1, 2))
    result = cp.empty((len(flattened), 3), dtype=cp.float64)
    launch(
        reference_kernel(),
        len(flattened),
        (flattened, result, len(flattened), end, substeps),
    )
    return (gaussian(result[:, :2], 0) * cp.exp(result[:, 2])).reshape(shape)


def validate():
    points = cp.random.RandomState(619).uniform(0, 1, (4096, 2))
    reference_error = float(
        cp.max(cp.abs(exact(points, 0.5) - exact(points, 0.5, 1024)))
    )
    assert reference_error < 3e-10, reference_error
    space = Space(Mesh(8), 4)
    reaction = Reaction(space)
    refined = Reaction(space, degree=20)
    quadrature_error = float(cp.max(cp.abs(reaction.matrix - refined.matrix)))
    assert quadrature_error < 3e-11, quadrature_error
    return dict(reference_error=reference_error, quadrature_error=quadrature_error)


def evolve(n=16, p=4, steps=16, modes=95, end=0.5, memory=True):
    space = Space(Mesh(n), p)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    carrier = FourierCarrier(space, modes=modes)
    transport = UnitaryCellular(modes)
    reaction = Reaction(space)
    g = reaction.exponential(end / (2 * steps))

    def physics(value):
        return cp.ascontiguousarray(cp.einsum("cij,cfj->cfi", g, value))

    initial = space.project(gaussian)
    state = MemoryTransport(carrier, initial)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(steps):
        state.apply_physics(physics)
        if not memory:
            state.auxiliary = carrier.project(state.value)
        state.advect(lambda value: transport.step(value, end / steps))
        state.apply_physics(physics)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    errors = space.errors(state.value, exact, end)
    ratio = float(cp.max(space.energy(state.value) / space.energy(initial)))
    assert ratio <= 1 + 1e-10, ratio
    return dict(
        n=n,
        p=p,
        steps=steps,
        modes=modes,
        end=end,
        memory=memory,
        energy_ratio=ratio,
        final_mass=float(space.mass(state.value)[0]),
        static_setup_seconds=setup,
        evolution_seconds=elapsed,
        **errors,
    )


def residual(n=16, p=4, modes=95):
    from flows import cellular_velocity

    space = Space(Mesh(n), p)
    carrier = FourierCarrier(space, modes=modes)
    transport = UnitaryCellular(modes)

    def derivative(points, t):
        value = gaussian(points, 0)
        gradient = (
            -12
            * cp.pi
            * cp.sin(2 * cp.pi * (points - cp.asarray([0.35, 0.45])))
            * value[..., None]
        )
        return -cp.sum(cellular_velocity(points, 0) * gradient, axis=-1)

    value = space.project(gaussian)
    a = space.project(derivative)
    ra = carrier.project(a)
    commutator = transport.generator(carrier.project(value)) - ra
    first = float(cp.sqrt(cp.mean(commutator**2)))
    norm = float(space.energy(a)[0])
    difference = norm - float(cp.mean(ra**2))
    assert difference >= -1e-12 * norm
    return dict(
        n=n,
        p=p,
        modes=modes,
        commutator_norm=first,
        defect_squared=max(0.0, difference),
        derivative_norm_squared=norm,
        defect_roundoff_scale=1e-13 * norm,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    validation = validate()
    meta = provenance()
    print(json.dumps(validation), flush=True)
    jobs = []
    for n in (8, 16, 32, 64):
        for p in (3, 4):
            jobs.append(("residual", dict(n=n, p=p)))
    for memory in (True, False):
        for n in (16, 32, 64):
            for steps in (4, 8, 16, 32, 64):
                jobs.append(("evolve", dict(n=n, steps=steps, memory=memory)))
    path = args.output / "reaction.jsonl"
    done = (
        {json.loads(l)["case_id"] for l in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    for kind, parameters in jobs:
        key = hashlib.sha256(
            json.dumps([kind, parameters], sort_keys=True).encode()
        ).hexdigest()[:16]
        if key in done:
            continue
        result = globals()[kind](**parameters)
        record = dict(
            status="ok",
            kind=kind,
            case_id=key,
            result=result,
            provenance=meta,
            validation=validation,
        )
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(dict(kind=kind, **result)), flush=True)
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
