"""Research checkpoint report for nonseparable transport and FE memory coupling.

Distributed under the Boost Software License, Version 1.0.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from report import Report, apply_style, plt, tex_escape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = (
        "day-validation",
        "day-cellular",
        "day-temporal",
        "day-enrichment",
        "day-nested-validation",
        "day-nested",
        "day-persistent",
        "day-spectral",
        "day-spectral-graph",
        "fourier-validation",
        "fourier-accuracy",
        "fourier-time-order",
        "fourier-autonomous",
        "fourier-compressed",
        "reaction",
    )
    raw = {
        name: [
            json.loads(line)
            for line in (args.input / (name + ".jsonl")).read_text().splitlines()
        ]
        for name in names
    }
    data = {
        name: [r["result"] for r in rows if r["status"] == "ok"]
        for name, rows in raw.items()
    }
    memory = json.loads((args.input / "memory-validation.json").read_text())
    failures = sum(r["status"] != "ok" for rows in raw.values() for r in rows)
    apply_style()
    r = Report(args.output)
    stamp = datetime.now(ZoneInfo("Europe/Paris")).strftime("%d/%m/%Y à %H:%M")
    r.paragraph(
        "Point d'étape de recherche du "
        + stamp
        + ". La branche SLFE contient des prototypes GPU "
        "distincts de l'interface de production Rodin. La campagne de nuit et le premier "
        "rapport incrémental restent disponibles séparément."
    )
    r.section("Résultat algorithmique")
    r.paragraph(
        "Le transport par projection statique est étendu à un flot cellulaire non séparable. "
        "Trois réalisations sont comparées : des supports DG orientés, un support DG unique "
        "avec changements de coordonnées incrémentaux, puis un support de Fourier unitaire. "
        "Le dernier est couplé aux triangles par deux projections adjointes FE–DG–Fourier."
    )
    r.paragraph(
        "Une différence structurelle apparaît entre le retour FE après chaque pas et la "
        "conservation d'un état auxiliaire. Le retour répété détruit en général l'ordre "
        "temporel élevé à espaces fixés. Conserver la mémoire auxiliaire permet de composer "
        "des sous-transports unitaires d'ordre quatre. Une règle de mise à jour de cette "
        "mémoire permet ensuite d'appliquer une opération physique directement dans l'espace FE."
    )
    r.tex.append(
        r"\[w^+=Sw,\quad v^+=v+R^*(w^+-w),\qquad v^\star=Gv,\quad w^\star=w+R(v^\star-v).\]"
    )
    r.body.append(
        "<p><strong>Advection :</strong> w⁺ = Sw, v⁺ = v + R*(w⁺ − w). "
        "<strong>Physique FE :</strong> v★ = Gv, w★ = w + R(v★ − v).</p>"
    )
    r.paragraph(
        "La note memory_theory.pdf établit une borne d'énergie dans un espace augmenté, "
        "l'ordre temporel du couplage à espaces fixés et des bornes conditionnelles de "
        "consistance. La racine du défaut de projection intervient dans la preuve, "
        "sans être calculée par l'algorithme à deux variables."
    )
    r.section("Champ, domaine et mesures")
    r.paragraph(
        "Le domaine est le tore unité. Le champ initial est une bosse gaussienne périodique. "
        "La fonction de courant est sin²(πx) sin²(πy)/π. Les essais utilisent soit sa vitesse "
        "stationnaire, soit cette vitesse multipliée par cos(πt). Les triangles sont perturbés "
        "et leurs diagonales varient. Les normes FE sont les normes L2 des polynômes reconstruits, "
        "avec une quadrature indépendante pour l'erreur. La solution de référence est obtenue "
        "par caractéristiques RK4 GPU, avec vérification par doublement de la résolution."
    )
    r.paragraph(
        "Les résultats principaux sont évalués à t = 0,5. Le retour à t = 1 du flot réversible "
        "est un contrôle de réversibilité ; il ne suffit pas à mesurer la précision du transport. "
        "Toutes les simulations, projections, décompositions spectrales et réductions numériques "
        "s'exécutent sur RTX 5090 en double précision. Symox prépare les polynômes exacts."
    )
    r.section("Contrôles")
    r.paragraph(
        f"{sum(map(len,raw.values()))} cas principaux sont enregistrés dans les JSONL, "
        f"avec {failures} échec d'exécution. S'y ajoutent quatre validations du couplage "
        "augmenté, six comparaisons des graphes CUDA avec les opérateurs ordinaires et "
        "huit validations FE–Fourier sous Compute Sanitizer."
    )
    sanitizer = (args.input / "sanitizer-fourier.log").read_text()
    r.paragraph(
        "Compute Sanitizer : "
        + " ; ".join(x.strip() for x in sanitizer.splitlines() if "ERROR SUMMARY" in x)
    )
    r.paragraph(
        "Les opérateurs de projection et les transports complets sont vérifiés par SVD "
        "dans la norme de masse sur de petits systèmes. Les adjoints sont comparés à une "
        "transposition dense indépendante. Les moments DG polynomiaux sont comparés à "
        "leur construction par quadrature. Les moments DG–Fourier sont recomputés avec "
        "48 points de Gauss au lieu de 32."
    )
    r.section("Convergence sur le flot cellulaire")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, label, steps in (
        ("day-cellular", "DG orientés, 64 pas", 64),
        ("day-nested", "DG imbriqué, 256 pas", 256),
        ("day-persistent", "DG persistant, 256 pas", 256),
    ):
        rows = sorted(
            [
                x
                for x in data[name]
                if x["p"] == 3
                and x["factor"] == 2
                and x["end"] == 0.5
                and x["steps"] == steps
            ],
            key=lambda x: x["n"],
        )
        ax.loglog(
            [1 / x["n"] for x in rows], [x["l2"][0] for x in rows], "o-", label=label
        )
    ax.set(
        xlabel=r"$h=1/n$",
        ylabel=r"$\|q_h-q\|_{L^2}$",
        title="P3, temps intermédiaire 0,5",
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=9)
    r.save_figure(
        fig,
        "cellular-spatial",
        "Le nombre de pas est indiqué pour chaque réalisation. Les courbes combinent erreur spatiale et erreur temporelle.",
    )
    rows = sorted(
        [
            x
            for x in data["fourier-compressed"]
            if x["modes"] == 95 and x["steps"] == 32
        ],
        key=lambda x: (x["p"], x["n"]),
    )
    r.table(
        ["p", "n", "DOF FE", "Erreur L2", "Évolution (s)", "Préparation (s)"],
        [
            [
                x["p"],
                x["n"],
                x["n"] ** 2 * (x["p"] + 1) * (x["p"] + 2),
                x["l2"][0],
                x["evolution_seconds"],
                x["static_setup_seconds"],
            ]
            for x in rows
        ],
    )
    r.paragraph(
        "Ces cas utilisent 95² modes auxiliaires et 32 pas d'ordre quatre avec mémoire. "
        "Le temps d'évolution comprend les échanges FE initiaux et finaux. La préparation "
        "comprend le couplage statique, les plans FFT, l'échauffement et la capture du graphe CUDA. "
        "Les nombres sont des mesures de campagne, sans intervalle statistique."
    )
    r.section("Comparaison à un solveur spectral Eulerien GPU")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparisons = (
        (
            "fourier-compressed",
            "FE–Fourier avec mémoire",
            lambda x: True,
            lambda x: x["l2"][0],
        ),
        (
            "day-spectral-graph",
            "Fourier/RK4, forme antisymétrique",
            lambda x: x["reversing"],
            lambda x: x["l2"],
        ),
    )
    for name, label, select, error in comparisons:
        rows = [x for x in data[name] if select(x)]
        ax.loglog(
            [x["evolution_seconds"] for x in rows],
            [error(x) for x in rows],
            "o",
            alpha=0.65,
            label=label,
        )
    ax.set(
        xlabel="Temps d'évolution (s)",
        ylabel=r"$\|q_h-q\|_{L^2}$",
        title="Même problème physique, préparation exclue",
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=9)
    r.save_figure(
        fig,
        "spectral-cost",
        "Les deux méthodes utilisent les graphes CUDA. Le solveur spectral exploite directement la géométrie périodique ; il ne fournit pas les mêmes inconnues FE.",
    )
    r.paragraph(
        "La référence Fourier/RK4 est compétitive et atteint de très petites erreurs avec "
        "peu d'inconnues sur ce problème lisse. Les essais ne démontrent pas une supériorité "
        "générale du couplage FE. Une comparaison avec un solveur DG direct de même degré, "
        "sur le même maillage et à erreur fixée, reste nécessaire."
    )
    r.section("Validation de la mémoire et de la physique FE")
    r.table(
        [
            "p",
            "Facteur",
            "Défaut isométrie",
            "Rapport erreur, λ=1",
            "Rapport erreur, λ=10",
        ],
        [
            [
                x["result"]["p"],
                x["result"]["factor"],
                x["result"]["isometry_error"],
                *[z["refinement_ratios"][-1] for z in x["result"]["results"]],
            ]
            for x in memory
        ],
    )
    r.paragraph(
        "Le rapport proche de quatre correspond au dernier doublement du nombre de pas "
        "d'une composition de Strang. La référence est l'exponentielle dense GPU du "
        "générateur augmenté. Les sous-transports sont exacts dans ce contrôle, afin "
        "d'isoler l'erreur de couplage. Un premier essai avec des sous-transports d'ordre "
        "quatre mélangeait les deux erreurs et n'atteignait pas encore ce régime asymptotique ; "
        "le diagnostic est conservé."
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in (16, 32, 64):
        rows = sorted(
            [x for x in data["reaction"] if x.get("memory") is True and x["n"] == n],
            key=lambda x: x["steps"],
        )
        ax.loglog(
            [0.5 / x["steps"] for x in rows],
            [x["l2"][0] for x in rows],
            "o-",
            label=f"P4, n = {n}",
        )
    ax.set(
        xlabel=r"$\Delta t$",
        ylabel=r"$\|q_h-q\|_{L^2}$",
        title="Advection et réaction FE spatiale",
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    r.save_figure(
        fig,
        "reaction-convergence",
        "Le puits vaut λ(x)=1+0,5 sin(2πx). Les exponentielles de réaction sont calculées dans chaque élément. La référence RK4 intègre indépendamment les caractéristiques et leur atténuation.",
    )
    r.paragraph(
        "Le test physique utilise une vitesse stationnaire et une réaction spatialement variable. "
        "La masse décroît physiquement ; la propriété contrôlée est la borne d'énergie, "
        "et l'erreur est évaluée contre la solution caractéristique. Les plafonds d'erreur "
        "aux petits pas diminuent avec le raffinement du maillage."
    )
    r.section("Portée scientifique")
    r.paragraph(
        "Les ingrédients ont des antécédents : transfert d'incréments FLIP, projections "
        "adjointes, dilatations d'opérateurs, transformations DG–Fourier et compositions "
        "de Yoshida. Aucune priorité scientifique n'est revendiquée à ce stade. La piste "
        "à approfondir est le couplage FE avec mémoire, son analyse de consistance et "
        "sa réalisation rapide, notamment lorsque plusieurs champs partagent le transport."
    )
    r.paragraph(
        "Les expériences actuelles couvrent un domaine périodique et une vitesse prescrite "
        "à quatre modes. Les frontières, les vitesses FE arbitraires, la compression, les "
        "flux locaux, le calcul distribué et une preuve générale de convergence restent ouverts. "
        "Les graphes CUDA accélèrent l'exécution sans changer les opérateurs numériques ; "
        "leurs résultats ont été comparés directement à ceux des exécutions ordinaires."
    )
    r.section("Reproduction")
    r.paragraph(
        "Les scripts day_study.py, fourier_study.py, spectral_reference.py, memory_study.py "
        "et reaction_study.py acceptent un répertoire de sortie explicite. Les JSONL "
        "contiennent les paramètres et les empreintes des sources. Les preuves sont dans "
        "incremental_theory.tex et memory_theory.tex. Aucun code de production Rodin "
        "ni fichier de la bibliothèque HySoP n'est modifié."
    )
    title = "SLFE : flot non séparable et couplage FE avec mémoire"
    style = "body{max-width:1050px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#152c3a}h2{margin-top:2em}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #cbd5de;padding:8px;text-align:left}img{max-width:100%}figcaption{font-size:14px}"
    (args.output / "rapport_journee.html").write_text(
        "<!doctype html><html lang='fr'><meta charset='utf-8'><title>"
        + title
        + "</title><style>"
        + style
        + "</style><h1>"
        + title
        + "</h1>"
        + "\n".join(r.body)
    )
    preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}\usepackage[french]{babel}
\usepackage{amsmath,amssymb,geometry,graphicx,booktabs,hyperref}
\geometry{margin=2cm}\hypersetup{colorlinks=true,urlcolor=blue,hypertexnames=false}
\setlength{\emergencystretch}{3em}
\title{SLFE : flot non séparable\\et couplage FE avec mémoire}
\author{Point d'étape de recherche GPU}\date{DATE}
\begin{document}\maketitle
""".replace(
        "DATE", tex_escape(stamp)
    )
    tex = preamble + "\n".join(r.tex) + r"\end{document}"
    tex = tex.replace("π", r"\(\pi\)").replace("λ", r"\(\lambda\)")
    (args.output / "rapport_journee.tex").write_text(tex)
    with (args.output / "pdf-build.log").open("w") as log:
        for _ in range(2):
            subprocess.run(
                [
                    "lualatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "rapport_journee.tex",
                ],
                cwd=args.output,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    summary = dict(
        generated=stamp,
        counts={name: len(rows) for name, rows in raw.items()},
        execution_failures=failures,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in args.output.glob("*.svg"):
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
