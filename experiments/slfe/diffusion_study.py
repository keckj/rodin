"""GPU SLFE coupling to implicit FE diffusion, with an analytic heat reference.

Distributed under the Boost Software License, Version 1.0.
The periodic translated heat kernel distinguishes spatial and time errors.
All assembly, Krylov iterations, reductions and dense control solves use CUDA.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Space, Mesh
from carrier import Carrier
from memory_transport import MemoryTransport
from fe_diffusion import FEDiffusion
from campaign import provenance


def heat_profile(points, time, fields=1, diffusivity=0.002):
    centers = cp.asarray([[0.35, 0.45], [0.64, 0.29], [0.2, 0.72], [0.76, 0.8]])[
        :fields
    ]
    sigma = cp.asarray([0.04, 0.035, 0.045, 0.05])[:fields]
    variance = sigma**2 + 2 * diffusivity * time
    delta = points[..., None, :] - centers - time * cp.asarray([1.0, 0.37])
    delta -= cp.rint(delta)
    gx, gy = 0.0, 0.0
    for shift in (-1, 0, 1):
        gx = gx + cp.exp(-((delta[..., 0] + shift) ** 2) / (2 * variance))
        gy = gy + cp.exp(-((delta[..., 1] + shift) ** 2) / (2 * variance))
    return sigma / (cp.sqrt(cp.pi) * variance) * gx * gy


def validate():
    records = []
    for degree in (1, 2, 4):
        space = Space(Mesh(4, jitter=0.18, seed=817), degree)
        diffusion = FEDiffusion(space)
        size = space.mesh.cells * space.nb
        identity = cp.ascontiguousarray(
            cp.eye(size).reshape(space.mesh.cells, space.nb, size).transpose(0, 2, 1)
        )
        matrix = diffusion.apply_scaled(identity).transpose(0, 2, 1).reshape(size, size)
        magnitude = float(cp.linalg.norm(matrix, ord=cp.inf))
        symmetry = float(cp.max(cp.abs(matrix - matrix.T))) / magnitude
        eigen, vectors = cp.linalg.eigh(matrix)
        minimum = float(eigen.min())
        one = cp.zeros((space.mesh.cells, 1, space.nb))
        one[:, :, 0] = 1
        null_error = (
            float(cp.max(cp.abs(diffusion.apply_scaled(one * diffusion.root_mass))))
            / magnitude
        )
        assert symmetry < 1e-12 and null_error < 1e-12 and minimum > -1e-11 * magnitude
        initial = cp.random.RandomState(381).normal(
            size=(space.mesh.cells, 2, space.nb)
        )
        duration = 0.03
        value = diffusion.step(initial, duration)
        weighted = (initial * diffusion.root_mass).transpose(0, 2, 1).reshape(size, 2)
        factor = duration * diffusion.diffusivity / 2
        exact = vectors @ (
            ((1 - factor * eigen) / (1 + factor * eigen))[:, None]
            * (vectors.T @ weighted)
        )
        actual = (value * diffusion.root_mass).transpose(0, 2, 1).reshape(size, 2)
        solve_error = float(cp.linalg.norm(actual - exact) / cp.linalg.norm(exact))
        mass_error = float(cp.max(cp.abs(space.mass(value) - space.mass(initial))))
        energy_ratio = float(cp.max(space.energy(value) / space.energy(initial)))
        assert solve_error < 1e-10 and mass_error < 1e-10 and energy_ratio <= 1 + 1e-10
        records.append(
            dict(
                degree=degree,
                symmetry_error=symmetry,
                constant_error=null_error,
                minimum_eigenvalue=minimum,
                matrix_norm=magnitude,
                dense_solve_error=solve_error,
                mass_error=mass_error,
                energy_ratio=energy_ratio,
            )
        )
    return records


def heat_only_controls():
    """Check the SIPG operator against an analytic Laplacian eigenfunction."""
    records = []
    diffusivity, duration, steps = 0.002, 0.2, 32

    def exact(points, time):
        return (
            cp.cos(2 * cp.pi * points[..., 0])
            * cp.sin(2 * cp.pi * points[..., 1])
            * cp.exp(cp.asarray(-8 * cp.pi**2 * diffusivity * time))
        )

    for n in (8, 16, 32):
        space = Space(Mesh(n, jitter=0.18, seed=817), 4)
        diffusion = FEDiffusion(space, diffusivity)
        value = space.project(exact, degree=20)
        for _ in range(steps):
            value = diffusion.step(value, duration / steps)
        error = space.errors(value, exact, duration, degree=20, subdivisions=2)
        floor = space.errors(
            space.project(exact, time=duration, degree=20),
            exact,
            duration,
            degree=20,
            subdivisions=2,
        )["l2"]
        records.append(
            dict(nfe=n, steps=steps, end=duration, best_fe_l2=floor, **error)
        )
    # A sign or normalization error in the stiffness would survive refinement.
    assert records[-1]["l2"][0] < 1e-7, records
    assert records[-1]["l2"][0] < records[0]["l2"][0] / 10, records
    return records


def evolve(nfe=16, steps=16, method="memory", fields=1, end=0.23, with_fields=False):
    space = Space(Mesh(nfe, jitter=0.18, seed=817), 4)
    profile = lambda points, time: heat_profile(points, time, fields)
    initial = space.project(profile, degree=20)
    cp.cuda.Stream.null.synchronize()
    before = time.perf_counter()
    diffusion = FEDiffusion(space)
    carrier = Carrier(space, n=2 * nfe)
    flow = carrier.transport(end / steps)
    state = MemoryTransport(carrier, initial)
    energy0, mass0 = state.energy(), space.mass(initial)
    diffusion.step(initial, end / (2 * steps))
    diffusion.solve_history.clear()
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - before
    before = time.perf_counter()
    history = []
    for step in range(steps):
        state.apply_physics(lambda v: diffusion.step(v, end / (2 * steps)))
        state.advect(flow.apply_carrier)
        if method == "reset":
            state.restart(carrier)
        elif method != "memory":
            raise ValueError("Expected memory or reset")
        state.apply_physics(lambda v: diffusion.step(v, end / (2 * steps)))
        energy_ratio = float(cp.max(state.energy() / energy0))
        mass_error = float(cp.max(cp.abs(space.mass(state.value) - mass0)))
        assert energy_ratio < 1 + 1e-9 and mass_error < 1e-9, (energy_ratio, mass_error)
        history.append(
            dict(step=step + 1, energy_ratio=energy_ratio, mass_error=mass_error)
        )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - before
    errors = space.errors(state.value, profile, end, degree=20, subdivisions=2)
    best = space.project(profile, time=end, degree=20)
    floor = space.errors(best, profile, end, degree=20, subdivisions=2)["l2"]
    record = dict(
        nfe=nfe,
        naux=2 * nfe,
        steps=steps,
        method=method,
        fields=fields,
        end=end,
        diffusivity=diffusion.diffusivity,
        setup_seconds=setup,
        evolution_seconds=elapsed,
        stiffness_builds=1,
        preconditioner_builds=len(diffusion.preconditioners),
        cg_solve_count=len(diffusion.solve_history),
        cg_iterations_max=max(x["iterations"] for x in diffusion.solve_history),
        cg_iterations_mean=sum(x["iterations"] for x in diffusion.solve_history)
        / len(diffusion.solve_history),
        cg_residual_max=max(x["relative_residual"] for x in diffusion.solve_history),
        history=history,
        best_fe_l2=floor,
        **errors
    )
    return (space, initial, state.value, profile, record) if with_fields else record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = provenance()
    records = validate()
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(
            dict(results=records, heat_only=heat_only_controls(), provenance=meta),
            indent=2,
        )
        + "\n"
    )
    cases = [
        dict(nfe=n, steps=s, method=m)
        for n in (8, 16, 32)
        for s in (4, 16, 64)
        for m in ("memory", "reset")
    ]
    cases += [dict(nfe=32, steps=16, fields=4, method=m) for m in ("memory", "reset")]
    with args.output.open("w") as stream:
        for case in cases:
            print(json.dumps(case), flush=True)
            result = evolve(**case)
            stream.write(json.dumps(dict(result=result, provenance=meta)) + "\n")
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
