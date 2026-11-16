"""Visualize the graded-carrier repair of the narrow-feature counterexample.

Distributed under the Boost Software License, Version 1.0.
Numerical sampling and error evaluation execute on CUDA.
"""

import argparse
import json
from pathlib import Path

from gpu import cp
from tensor_study import evolve, partitions
from field_report import sample, save
from report import plt, apply_style
from matplotlib.colors import LogNorm, Normalize
from campaign import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    output = args.input / "visuals"
    apply_style()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    space, result, initial, exact, record = evolve(
        n=128, steps=4, flow="cellular", with_fields=True
    )
    extent = (0.338, 0.362, 0.434, 0.460)
    x = cp.linspace(*extent[:2], 512)
    y = cp.linspace(*extent[2:], 512)
    yy, xx = cp.meshgrid(y, x, indexing="ij")
    points = cp.stack((xx, yy), axis=-1)
    value = sample(space, result, points)[..., 0]
    reference = exact(points, 0)
    error = cp.abs(value - reference)
    reference, value, error = reference.get(), value.get(), error.get()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), layout="constrained")
    common = Normalize(
        vmin=min(float(value.min()), 0.0),
        vmax=max(float(value.max()), float(reference.max())),
    )
    for ax, field, title, norm, cmap in zip(
        axes,
        (reference, value, error),
        (
            "Référence par caractéristiques",
            "Auxiliaire gradué\n" + rf"$\|e\|_{{L^2}}={record['l2'][0]:.3g}$",
            "Erreur absolue",
        ),
        (common, common, LogNorm(vmin=1e-5, vmax=max(float(error.max()), 1e-4))),
        ("viridis", "viridis", "magma"),
    ):
        plot = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            norm=norm,
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set(xlabel="x", ylabel="y", title=title)
        ax.set_aspect("equal")
        fig.colorbar(plot, ax=ax, shrink=0.78)
    fig.suptitle(
        "La bosse fine est transportée avec le support gradué\nMême flot cellulaire et mêmes triangles P4 — 128 × 128 cellules Q4 auxiliaires, 4 pas",
        fontsize=14,
    )
    save(fig, output, "fields-graded")
    (output / "fields-graded.json").write_text(json.dumps(record, indent=2) + "\n")

    rows = [
        json.loads(line)["result"]
        for line in (args.input / "tensor-cellular.jsonl").read_text().splitlines()
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for n, color in ((64, "#216b86"), (128, "#bd6536")):
        selected = [r for r in rows if r["n"] == n]
        axes[0].loglog(
            [r["steps"] for r in selected],
            [r["l2"][0] for r in selected],
            "o-",
            color=color,
            label=f"{n} × {n} cellules Q4",
        )
    axes[0].axhline(
        record["best_fe_l2"][0],
        color="#555555",
        linestyle="--",
        label="Meilleure projection FE",
    )
    axes[0].set(
        xlabel="Nombre de pas pour t = 0,005",
        ylabel=r"Erreur $L^2$",
        title="La limite est désormais spatiale",
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2, which="both")
    edges = partitions(64, 0.95).get()
    for axis in (0, 1):
        for e in edges[axis]:
            if axis == 0:
                axes[1].axvline(e, linewidth=0.65, color="#447b47")
            else:
                axes[1].axhline(e, linewidth=0.65, color="#447b47")
    axes[1].add_patch(
        plt.Circle((0.35, 0.45), 0.0015, fill=False, edgecolor="#c84a32", linewidth=2)
    )
    axes[1].set(
        xlim=(0.33, 0.37),
        ylim=(0.43, 0.47),
        xlabel="x",
        ylabel="y",
        title="Support gradué, 64 × 64 cellules",
    )
    axes[1].set_aspect("equal")
    save(fig, output, "graded-resolution")
    (output / "tensor-visual-provenance.json").write_text(
        json.dumps(provenance(), indent=2) + "\n"
    )
    for name in ("fields-graded", "graded-resolution"):
        p = output / (name + ".svg")
        p.write_text(
            "\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n"
        )
    print(json.dumps(record))


if __name__ == "__main__":
    main()
