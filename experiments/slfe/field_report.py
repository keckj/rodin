"""Plot reconstructed FE fields, independent characteristics and pointwise errors.

Distributed under the Boost Software License, Version 1.0.
Numerical solves and sampling run on CUDA. Matplotlib only renders results.
"""

import argparse
import json
import math
from pathlib import Path
import time

from gpu import cp, HERE, Space, gaussian, launch
from graded_study import GradedMesh
from fourier_carrier import FourierCarrier, UnitaryCellular
from eulerian_dg import EulerianDG
from gpu_graph import EvolutionGraph
from day_study import reference_feet
from campaign import provenance
from report import plt, apply_style
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection


def sample(space, coefficients, points, grading=0.95):
    source = f"#define NB {space.nb}\n#define M_SQRT2 1.41421356237309504880\n"
    source += (
        (HERE / "basis.cuh").read_text()
        + (HERE / "transfer.cu").read_text()
        + (HERE / "field_sample.cu").read_text()
    )
    kernel = cp.RawKernel(source, "sample_fields")
    flat = cp.ascontiguousarray(points.reshape(-1, 2))
    result = cp.empty((len(flat), coefficients.shape[1]))
    owners = cp.empty(len(flat), dtype=cp.int32)
    launch(
        kernel,
        len(flat),
        (
            flat,
            len(flat),
            space.mesh.triangles,
            space.mesh.n,
            grading,
            coefficients,
            coefficients.shape[1],
            result,
            owners,
        ),
    )
    assert bool(cp.all(owners >= 0)), int(cp.sum(owners < 0))
    return result.reshape(*points.shape[:-1], coefficients.shape[1])


def solve(narrow=False):
    mesh = GradedMesh(64, 0.95)
    space = Space(mesh, 4)
    width = 0.0015
    if narrow:

        def profile(points, t):
            r = cp.sin(cp.pi * (points - cp.asarray([0.35, 0.45])))
            return (
                cp.exp(-cp.sum(r * r, axis=-1) / (2 * (cp.pi * width) ** 2))
                / (math.sqrt(math.pi) * width)
            )[..., None]

        duration = 0.005
        extent = (0.338, 0.362, 0.434, 0.460)
    else:
        profile = gaussian
        duration = 1 / math.pi
        extent = (0.0, 1.0, 0.0, 1.0)
    initial = space.project(profile, degree=20)

    def exact(points, t):
        return profile(reference_feet(points, -duration, 512), 0)

    outcomes = {}
    coefficients = [initial]
    for method in ("auxiliary", "dg"):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        if method == "auxiliary":
            carrier = FourierCarrier(space, modes=95)
            flow = UnitaryCellular(95)
            steps = 16
            old = carrier.project(initial)
            graph = EvolutionGraph(flow.step, old, duration / steps)
        else:
            flow = EulerianDG(space, flux="upwind")
            scale = 6 if narrow else 8
            steps = math.ceil(duration * flow.norm_bound / (2 * scale))
            graph = EvolutionGraph(flow.step, initial, duration / steps)
        cp.cuda.Stream.null.synchronize()
        setup = time.perf_counter() - start
        start = time.perf_counter()
        if method == "auxiliary":
            old = carrier.project(initial)
            graph.reset(old)
        for _ in range(steps):
            graph.launch(duration / steps)
        value = graph.result()
        if method == "auxiliary":
            value = initial + carrier.adjoint(value - old)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        errors = space.errors(
            value, exact, 0, degree=20, subdivisions=2 if narrow else 1
        )
        outcomes[method] = dict(
            steps=steps, setup_seconds=setup, evolution_seconds=elapsed, **errors
        )
        coefficients.append(value)
    size = 512
    x = cp.linspace(extent[0], extent[1], size)
    y = cp.linspace(extent[2], extent[3], size)
    yy, xx = cp.meshgrid(y, x, indexing="ij")
    points = cp.stack((xx, yy), axis=-1)
    sampled = sample(space, cp.concatenate(coefficients, axis=1), points)
    reference = exact(points, 0)
    if reference.ndim == 3:
        reference = reference[..., 0]
    pointwise_errors = cp.stack(
        (cp.abs(sampled[..., 1] - reference), cp.abs(sampled[..., 2] - reference)),
        axis=-1,
    )
    sampled = cp.concatenate((sampled, pointwise_errors), axis=-1)
    # Independently check evaluation at known element-interior coordinates.
    coords = cp.asarray([[0.2, 0.2], [0.6, 0.2], [0.2, 0.6]])
    test_points = mesh.points(cp.column_stack((coords, cp.ones(3))))
    direct = cp.einsum(
        "cfb,qb->cqf", cp.concatenate(coefficients, axis=1), space.basis(coords)
    )
    located = sample(space, cp.concatenate(coefficients, axis=1), test_points)
    discrepancy = float(cp.max(cp.abs(direct - located)))
    assert discrepancy < 2e-8, discrepancy
    return (
        sampled.get(),
        reference.get(),
        mesh.triangles.get(),
        dict(
            narrow=narrow,
            n=64,
            p=4,
            grading=0.95,
            extent=extent,
            physical_end=0.005 if narrow else 0.5,
            transport_duration=duration,
            width=width if narrow else None,
            methods=outcomes,
            sampling_check=discrepancy,
        ),
    )


def save(fig, output, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            output / (name + "." + ext),
            dpi=160,
            bbox_inches="tight",
            transparent=ext != "png",
            facecolor="white" if ext == "png" else "none",
        )
    plt.close(fig)


def fields(output, narrow=False):
    values, reference, triangles, meta = solve(narrow)
    name = "fields-narrow" if narrow else "fields-smooth"
    initial, auxiliary, dg = [values[:, :, i] for i in range(3)]
    import numpy as np

    # Host operations below concern plot ranges and rendering only.
    vmin = min(0.0, float(values.min()))
    vmax = max(float(values.max()), float(reference.max()))
    normal = Normalize(vmin=vmin, vmax=vmax)
    error_aux = values[..., 3]
    error_dg = values[..., 4]
    errmax = max(float(error_aux.max()), float(error_dg.max()))
    errmin = 1e-5 if narrow else 1e-10
    errornorm = LogNorm(vmin=errmin, vmax=max(errmax, 10 * errmin))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), layout="constrained")
    extent = meta["extent"]
    panels = [
        (initial, "Initial : projection FE", normal, "viridis"),
        (reference, "Référence par caractéristiques", normal, "viridis"),
        (
            auxiliary,
            f"Auxiliaire, 95² valeurs\nErreur L2 = {meta['methods']['auxiliary']['l2'][0]:.3g}",
            normal,
            "viridis",
        ),
        (
            dg,
            f"DG amont\nErreur L2 = {meta['methods']['dg']['l2'][0]:.3g}",
            normal,
            "viridis",
        ),
        (
            np.maximum(error_aux, errmin),
            "Erreur absolue auxiliaire",
            errornorm,
            "magma",
        ),
        (np.maximum(error_dg, errmin), "Erreur absolue DG", errornorm, "magma"),
    ]
    for ax, (array, title, norm, cmap) in zip(axes.ravel(), panels):
        picture = ax.imshow(
            array,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            norm=norm,
            cmap=cmap,
        )
        ax.set(title=title, xlabel="x", ylabel="y")
        ax.set_aspect("equal")
        color = fig.colorbar(picture, ax=ax, shrink=0.83, pad=0.025)
        color.ax.tick_params(labelsize=9)
    title = (
        "Petite échelle : le support auxiliaire manque la bosse"
        if narrow
        else "Champ lisse : les deux méthodes transportent la même bosse"
    )
    subtitle = f"8192 triangles P4 gradués · t = {meta['physical_end']} · mêmes échelles de couleur pour les champs et pour les erreurs"
    fig.suptitle(title + "\n" + subtitle, fontsize=14)
    save(fig, output, name)
    (output / (name + ".json")).write_text(json.dumps(meta, indent=2) + "\n")
    return meta, triangles


def diagram(output):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set(xlim=(0, 12), ylim=(0, 5.5))
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.15",
                facecolor=color,
                edgecolor="#486578",
                linewidth=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12)

    def arrow(a, b, label, offset=(0, 0)):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=17,
                color="#38546b",
                linewidth=1.5,
            )
        )
        ax.text(
            (a[0] + b[0]) / 2 + offset[0],
            (a[1] + b[1]) / 2 + offset[1],
            label,
            ha="center",
            va="center",
            fontsize=11,
        )

    box(
        0.35,
        2.6,
        3.35,
        1.5,
        "Éléments finis\nTriangles P4, coefficients v\nRéaction / physique FE",
        "#e2edf4",
    )
    box(
        8.1,
        2.6,
        3.35,
        1.5,
        "État auxiliaire persistant\nValeurs w sur grille régulière\nTransport par cisaillements FFT",
        "#e7f1e5",
    )
    arrow((3.9, 3.8), (7.9, 3.8), "Initialisation : w = Rv", (0, 0.35))
    arrow((7.9, 3.05), (3.9, 3.05), r"Retour : $v\leftarrow v+R^*(w^+-w)$", (0, -0.33))
    box(
        3.1,
        0.4,
        5.7,
        1.05,
        "Après une opération physique FE :\n" + r"$w\leftarrow w+R(v^+-v)$",
        "#fbefd9",
    )
    arrow((2.0, 2.4), (3.6, 1.65), "", (-0.1, 0))
    arrow((8.5, 1.65), (9.8, 2.4), "")
    ax.text(
        6,
        4.95,
        "Ce qui est effectivement calculé",
        ha="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        6,
        4.5,
        "R : projection statique FE → DG cartésien → Fourier ; R* : adjoint dans les normes de masse",
        ha="center",
        fontsize=10,
    )
    save(fig, output, "method-diagram")


def costs(output, smooth, narrow):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, case, title in zip(axes, (smooth, narrow), ("Champ lisse", "Bosse fine")):
        entries = [case["methods"][m] for m in ("dg", "auxiliary")]
        for i, (label, color) in enumerate(
            (("Préparation", "#9cb5c5"), ("Évolution", "#216b86"))
        ):
            key = "setup_seconds" if i == 0 else "evolution_seconds"
            ax.barh(
                [0, 1],
                [x[key] for x in entries],
                left=[0, 0] if i == 0 else [x["setup_seconds"] for x in entries],
                label=label,
                color=color,
            )
        ax.set_yticks([0, 1], ["DG amont", "Auxiliaire 95²"])
        ax.invert_yaxis()
        ax.set(xlabel="Temps (s)", title=title)
        for j, x in enumerate(entries):
            ax.text(
                0.02,
                j - 0.31,
                f"Erreur L2 = {x['l2'][0]:.3g}",
                transform=ax.get_yaxis_transform(),
                fontsize=10,
                color="#172f40",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
            )
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    fig.suptitle(
        "Coût total et précision : le gain dépend du champ transporté", fontsize=15
    )
    save(fig, output, "visible-cost")


def mesh_picture(output, triangles):
    import numpy as np

    bounds = (0.33, 0.37, 0.43, 0.47)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    edges = np.stack((triangles, np.roll(triangles, -1, axis=1)), axis=2).reshape(
        -1, 2, 2
    )
    axes[0].add_collection(LineCollection(edges, colors="#216b86", linewidths=0.45))
    for value in np.arange(96) / 95:
        axes[1].axvline(value, color="#447b47", linewidth=1)
        axes[1].axhline(value, color="#447b47", linewidth=1)
    for ax, title in zip(
        axes,
        (
            "Triangles FE dans la zone raffinée",
            "Grille auxiliaire à 95 points par direction",
        ),
    ):
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="x", ylabel="y", title=title)
        ax.set_aspect("equal")
        ax.add_patch(
            plt.Circle(
                (0.35, 0.45), 0.0015, fill=False, edgecolor="#c84a32", linewidth=2
            )
        )
    fig.suptitle(
        "Même zoom physique — cercle rouge : largeur de la bosse fine", fontsize=15
    )
    save(fig, output, "mesh-vs-carrier")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    apply_style()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    smooth, triangles = fields(args.output)
    print("smooth complete", flush=True)
    narrow, _ = fields(args.output, True)
    print("narrow complete", flush=True)
    diagram(args.output)
    mesh_picture(args.output, triangles)
    costs(args.output, smooth, narrow)
    (args.output / "visual-provenance.json").write_text(
        json.dumps(provenance(), indent=2) + "\n"
    )
    for path in args.output.glob("*.svg"):
        path.write_text(
            "\n".join(l.rstrip() for l in path.read_text().splitlines()) + "\n"
        )


if __name__ == "__main__":
    main()
