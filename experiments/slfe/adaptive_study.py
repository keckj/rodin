"""Solution-driven graded carriers with contractive memory restarts.

Distributed under the Boost Software License, Version 1.0.
Circular energy centroids are GPU-computed heuristics for one localized feature.
Restarts preserve the FE value and discard a measured orthogonal memory part.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Space, quadrature
from graded_study import GradedMesh
from carrier import Carrier
from tensor_cellular import TensorCellular
from memory_transport import MemoryTransport
from day_study import reference_feet
from campaign import provenance


def partitions(n, grading, center):
    t = cp.linspace(0.0, 1.0, n + 1)
    return cp.ascontiguousarray(
        t[None, :]
        + grading
        / (2 * cp.pi)
        * cp.sin(2 * cp.pi * (t[None, :] + 0.5 - center[:, None]))
    )


def fe_center(space, coefficients):
    q = quadrature(16)
    values = cp.einsum("cfb,qb->cqf", coefficients, space.basis(q[:, :2]))
    weights = cp.sum(values * values, axis=2) * (
        2 * space.mesh.areas[:, None] * q[None, :, 2]
    )
    moments = cp.sum(
        weights[..., None] * cp.exp(2j * cp.pi * space.mesh.points(q)), axis=(0, 1)
    )
    return (cp.angle(moments) / (2 * cp.pi)) % 1


def tensor_center(carrier, values):
    c = carrier
    weights = cp.sum(values * values, axis=1) * c.mass
    weights = weights.reshape(c.n, c.n, c.np, c.np)
    marginal = [cp.sum(weights, axis=(0, 2)), cp.sum(weights, axis=(1, 3))]
    positions = c.edges[:, :-1, None] + cp.diff(c.edges, axis=1)[:, :, None] * c.nodes
    moments = cp.stack(
        [cp.sum(marginal[a] * cp.exp(2j * cp.pi * positions[a])) for a in (0, 1)]
    )
    concentration = cp.abs(moments) / cp.sum(weights)
    return (cp.angle(moments) / (2 * cp.pi)) % 1, concentration


def evolve(
    nfe=64,
    naux=32,
    width=0.01,
    end=0.1,
    steps=64,
    check_every=4,
    threshold=0.03,
    grading=0.95,
    adaptive=True,
    restart_same=False,
    carry_memory=False,
    projection_iterations=0,
    filter_scale=None,
    reaction=False,
    fields=1,
    spread=0.008,
    mesh_policy="centroid",
    with_fields=False,
):
    space = Space(GradedMesh(nfe, 0.95), 4)

    from moving_study import profile_family

    profile = profile_family(width, fields, spread)

    def exact(points, t):
        if reaction:
            from reaction_study import reference_kernel
            from gpu import launch

            flat = cp.ascontiguousarray(points.reshape(-1, 2))
            traced = cp.empty((len(flat), 3))
            launch(reference_kernel(), len(flat), (flat, traced, len(flat), end, 512))
            return (profile(traced[:, :2], 0) * cp.exp(traced[:, 2, None])).reshape(
                *points.shape[:-1], fields
            )
        return profile(reference_feet(points, -end, 512), 0)

    initial = space.project(profile, degree=20)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    center = fe_center(space, initial)
    original_center = center.copy()
    oriented = None
    if mesh_policy == "energy":
        from energy_mesh import from_fe, from_tensor

        edges, oriented = from_fe(space, initial, naux)
    elif mesh_policy == "centroid":
        edges = partitions(naux, grading, center)
    else:
        raise ValueError("Expected centroid or energy mesh policy")
    carrier = Carrier(space, n=naux, edges=edges)
    flow = TensorCellular(
        carrier, end / steps, grading, center=center, oriented_edges=oriented
    )
    state = MemoryTransport(carrier, initial)
    physics = None
    if reaction:
        from reaction_study import Reaction

        operator = Reaction(space, degree=20)
        matrix = operator.exponential(end / (2 * steps))
        physics = lambda v: operator.apply(matrix, v)
    cp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    initial_energy = space.energy(initial)
    history = []
    rebuild = 0.0
    start = time.perf_counter()
    for first in range(0, steps, check_every):
        count = min(check_every, steps - first)

        def operation(w):
            for _ in range(count):
                w = flow.apply_carrier(w)
            return w

        if reaction:
            for _ in range(count):
                state.apply_physics(physics)
                state.advect(flow.apply_carrier)
                state.apply_physics(physics)
        else:
            state.advect(operation)
        detected, concentration = tensor_center(carrier, state.auxiliary)
        displacement = (detected - center + 0.5) % 1 - 0.5
        distance = float(cp.linalg.norm(displacement))
        change = adaptive and first + count < steps and distance > threshold
        next_edges, next_oriented = None, None
        if mesh_policy == "energy":
            next_edges, next_oriented = from_tensor(carrier, state.auxiliary)
            movement = max(
                float(cp.max(cp.abs(next_edges - carrier.edges))),
                float(cp.max(cp.abs(next_oriented - oriented))),
            )
            change = adaptive and first + count < steps and movement > threshold / 3
        loss = None
        if change:
            # The centroid may cease to describe a single feature when energy
            # spreads around the torus. Such cases are retained, not hidden.
            cp.cuda.Stream.null.synchronize()
            before = time.perf_counter()
            new_center = original_center if restart_same else detected
            new_carrier = Carrier(
                space,
                n=naux,
                edges=(
                    next_edges
                    if next_edges is not None
                    else partitions(naux, grading, new_center)
                ),
            )
            if carry_memory or projection_iterations:
                from tensor_grid import TensorGrid

                target = TensorGrid(new_carrier, new_carrier.edges)
                remap = lambda w: target.remap(carrier, w)
                if projection_iterations:
                    loss = state.project_memory(
                        new_carrier, remap, projection_iterations, filter_scale
                    )
                else:
                    loss = state.transfer_memory(new_carrier, remap)
            else:
                loss = state.restart(new_carrier)
            carrier = new_carrier
            oriented = next_oriented
            flow = TensorCellular(
                carrier,
                end / steps,
                grading,
                center=new_center,
                oriented_edges=oriented,
            )
            center = new_center.copy()
            cp.cuda.Stream.null.synchronize()
            rebuild += time.perf_counter() - before
        mass_error = float(
            cp.max(cp.abs(space.mass(state.value) - space.mass(initial)))
        )
        energy_ratio = float(cp.max(state.energy() / initial_energy))
        assert (reaction or mass_error < 2e-9) and energy_ratio < 1 + 2e-9, (
            mass_error,
            energy_ratio,
        )
        history.append(
            dict(
                step=first + count,
                center=center.tolist(),
                detected=detected.tolist(),
                concentration=concentration.tolist(),
                displacement=distance,
                restarted=change,
                memory_loss=None if loss is None else loss.tolist(),
                mass_error=None if reaction else mass_error,
                mass=space.mass(state.value).tolist(),
                energy_ratio=energy_ratio,
            )
        )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    errors = space.errors(state.value, exact, 0, degree=20, subdivisions=2)
    best = space.errors(
        space.project(exact, degree=20), exact, 0, degree=20, subdivisions=2
    )["l2"]
    result = dict(
        nfe=nfe,
        naux=naux,
        width=width,
        end=end,
        steps=steps,
        threshold=threshold,
        check_every=check_every,
        adaptive=adaptive,
        restart_same=restart_same,
        carry_memory=carry_memory,
        projection_iterations=projection_iterations,
        filter_scale=filter_scale,
        reaction=reaction,
        fields=fields,
        spread=spread,
        mesh_policy=mesh_policy,
        grading=grading,
        setup_seconds=setup,
        evolution_seconds=elapsed,
        rebuild_seconds=rebuild,
        restarts=sum(r["restarted"] for r in history),
        best_fe_l2=best,
        history=history,
        **errors
    )
    return (space, initial, state.value, exact, result) if with_fields else result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=(
            "smoke",
            "resolution",
            "threshold",
            "memory",
            "projected",
            "physics",
            "chebyshev",
            "separated",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.suite == "smoke":
        cases = [dict(adaptive=a) for a in (False, True)]
    elif args.suite == "resolution":
        cases = [
            dict(nfe=nfe, naux=naux, end=end, steps=steps, adaptive=a)
            for nfe in (64, 128)
            for naux in (16, 32, 64)
            for end, steps in ((0.1, 64), (0.25, 128))
            for a in (False, True)
        ]
    elif args.suite == "memory":
        cases = [
            dict(nfe=nfe, naux=64, end=0.25, steps=128, carry_memory=m)
            for nfe in (64, 128)
            for m in (False, True)
        ]
    elif args.suite == "projected":
        cases = [
            dict(nfe=nfe, naux=64, end=0.25, steps=128, projection_iterations=k)
            for nfe in (64, 128)
            for k in (1, 2, 4, 8, 16)
        ]
    elif args.suite == "physics":
        cases = [
            dict(nfe=nfe, naux=64, end=0.25, steps=steps, reaction=True, **method)
            for nfe in (64, 128)
            for steps in (64, 128, 256)
            for method in (
                dict(),
                dict(carry_memory=True),
                dict(projection_iterations=1),
                dict(projection_iterations=8),
            )
        ]
    elif args.suite == "chebyshev":
        cases = [
            dict(
                nfe=nfe,
                naux=64,
                end=0.25,
                steps=128,
                projection_iterations=k,
                filter_scale=s,
            )
            for nfe in (64, 128)
            for s in (0.01, 0.1)
            for k in (2, 4, 8)
        ]
    elif args.suite == "separated":
        cases = [
            dict(
                nfe=nfe,
                naux=64,
                end=0.25,
                steps=128,
                fields=4,
                spread=0.15,
                mesh_policy=policy,
                reaction=react,
                projection_iterations=k,
            )
            for nfe in (128, 256)
            for react in (False, True)
            for policy, k in (("centroid", 8), ("energy", 0), ("energy", 8))
        ]
    else:
        cases = [
            dict(nfe=128, naux=32, end=0.25, steps=128, threshold=t, restart_same=s)
            for t in (0.01, 0.03, 0.06)
            for s in (False, True)
        ]
    meta = provenance()
    with args.output.open("w") as stream:
        for parameters in cases:
            print(json.dumps(parameters), flush=True)
            result = evolve(**parameters)
            stream.write(
                json.dumps(dict(parameters=parameters, result=result, provenance=meta))
                + "\n"
            )
            stream.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
