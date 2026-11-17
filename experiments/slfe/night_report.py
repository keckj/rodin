"""Illustrated report of adaptive-support SLFE experiments.

Distributed under the Boost Software License, Version 1.0.
Host-only analysis of recorded GPU results; no numerical simulation here.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import statistics
import subprocess
from zoneinfo import ZoneInfo

from report import Report, load, plt, apply_style, tex_escape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    apply_style()
    plt.rcParams["svg.hashsalt"] = "slfe-night2"
    r = Report(args.output)
    stamp = datetime.now(ZoneInfo("Europe/Paris")).strftime("%d/%m/%Y à %H:%M")
    raw = {
        name: load(args.input, name)
        for name in (
            "matched",
            "adaptive-memory",
            "adaptive-projected",
            "adaptive-physics",
            "adaptive-chebyshev",
            "moving-separated",
            "adaptive-separated",
            "memory-stress",
            "separated-refinement",
            "roundtrip-smoke",
            "roundtrip-long",
            "auxiliary-return",
            "source-smoke",
            "source-full",
            "source-refinement",
            "product-pilot",
            "products",
            "monitor-stationary",
            "monitor-return",
            "svd",
            "compressible",
            "warm-return",
            "material-monitor",
            "material-phase",
            "monitor-fe",
            "phase-return",
            "galerkin",
            "budgeted-return",
            "budgeted-remaining",
            "joint-return",
            "frozen-transfer",
            "krylov",
            "krylov-tuned",
            "krylov-fine",
            "block-frozen",
            "gram",
            "fast-return",
            "system-coupling",
            "mixture",
            "relaxation",
            "reaction-moments",
            "reaction-moments-controls",
            "reaction-moments-fine",
            "reaction-moments-quadrature",
            "reaction-projection",
            "spectral-reaction",
            "tiled-return-pilot",
            "long-return-benchmark",
            "residual-return",
            "reaction-increment-projection",
            "reaction-residual-projection",
            "increment-physical",
            "increment-reanchor",
            "increment-conditioning",
            "increment-conditioning-positive",
            "graph-return",
            "graph-return-long",
            "graph-frozen",
            "remap-consistency",
            "remap-consistency-orders",
            "remap-consistency-fine",
            "remap-consistency-smallstep",
            "remap-consistency-interval",
            "material-minimax",
            "minimax-return",
            "diffusion",
            "moving-resolution",
            "moving-fields",
        )
    }
    r.paragraph(
        "Résultats disponibles le "
        + stamp
        + " (Europe/Paris). Calculs en double précision sur une RTX 5090. Les références sont obtenues par caractéristiques indépendantes, par retour exact du flot ou par solution analytique, selon le test."
    )
    r.section("Résultat et portée")
    r.paragraph(
        "Trois résultats structurent désormais l'étude : une identité d'énergie relative autorise une borne d'erreur malgré les transferts non linéaires; les réactions appliquées au champ complet évitent une mémoire artificiellement persistante; une réalisation linéaire par réflexion orthogonale conserve une précision P4 proche de l'ordre cinq dans le test de translation à changements fréquents. Le rapport détaille aussi les cas où ces méthodes perdent en coût ou en précision."
    )
    r.body.append(
        "<p><a href='memory_paper_draft.pdf'>Ébauche d'article avec preuves et expériences principales (PDF)</a></p>"
    )
    r.tex.append(
        r"\noindent\href{memory_paper_draft.pdf}{Ébauche d'article avec preuves et expériences principales.}\par"
    )
    r.paragraph(
        "Un support auxiliaire mobile résout le défaut observé lorsque la bosse quitte la zone fine d'une grille fixe. Le couplage FE avec mémoire accepte désormais un changement de support contractant, et un raffinement exactement emboîté sans perte d'énergie. Les transferts polynomiaux évitent une inversion globale; la variante de Galerkin diagonalise une petite matrice par champ. Les propriétés sont établies sous les hypothèses de restart_theory.pdf; elles n'établissent ni une nouveauté bibliographique, ni une convergence générale vers l'EDP."
    )
    r.paragraph(
        "Deux méthodes sont distinguées : le couplage avec mémoire maintient le champ FE à chaque changement; la référence d'advection pure transporte uniquement le champ DG auxiliaire entre une projection FE initiale et une projection FE finale. Cette seconde méthode est la plus rapide ici, mais elle n'offre pas les opérations physiques FE intermédiaires du couplage."
    )
    r.paragraph(
        "Les contre-tests exposent deux limites importantes : un transfert itératif tronqué peut dissiper même à support inchangé, et un contrôle de maillage tous les quatre pas modifie le problème d'adaptation lorsque le pas diminue. Une variante corrige le premier défaut au prix d'une non-linéarité; une ablation à intervalle physique fixé isole le second. La composition hamiltonienne par SVD perd la comparaison de coût et de précision contre Fourier/RK4 sur le vortex résolu. Ces résultats orientent la suite de la recherche."
    )
    if raw["reaction-moments"]:
        r.paragraph(
            "Le résultat nouveau de la fin de nuit concerne la physique de la mémoire : une contraction de l'énergie ne suffit pas à reproduire une réaction forte. Le prototype applique désormais la réaction spatiale au champ complet, puis conserve exactement ses moments FE et comprime sa composante cachée dans trois directions. Sur le cas de réaction forte testé, l'erreur relative passe de 32,7 pour un amortissement uniforme de la mémoire à 0,00240 pour ces moments complets, avec un plancher FE de 0,00237. C'est une correction de modèle démontrée sur ce test, à un coût supérieur; ce résultat ne constitue pas à lui seul une avancée bibliographique établie."
        )
    for path in (args.input / "figures").glob("*"):
        if path.suffix in (".svg", ".pdf", ".png", ".json"):
            shutil.copy2(path, args.output / path.name)
    r.section("Construction étudiée")
    r.paragraph(
        "Le maillage triangulaire FE reste fixe et porte les opérations de physique. Un second maillage transporte le champ par cisaillements semi-lagrangiens. Après ce transport, le champ FE reçoit l'incrément projeté; l'état auxiliaire est conservé pour le pas suivant. Cette mémoire évite de reconstruire entièrement le champ transporté à chaque pas. Le problème étudié cette nuit est de modifier le support auxiliaire en préservant le champ FE, la masse et un bilan d'énergie contrôlé."
    )
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set(xlim=(0, 10), ylim=(0, 3.5))
    ax.axis("off")
    boxes = (
        (
            0.1,
            1.6,
            3.0,
            1.5,
            "Physique sur les triangles FE",
            r"$v^+=G(v)$" + "\nDiffusion, réaction ou source",
        ),
        (
            6.9,
            1.6,
            3.0,
            1.5,
            "Transport et mémoire auxiliaire",
            r"$w^+=S(w)$" + "\nÉtat conservé entre les pas",
        ),
        (
            2.5,
            0.05,
            5.0,
            0.8,
            "Changement du support auxiliaire",
            "Préserver le champ FE et contrôler la mémoire transférée",
        ),
    )
    for x, y, width, height, title, body in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.04",
                facecolor="#eef3f6",
                edgecolor="#52728b",
            )
        )
        ax.text(
            x + width / 2,
            y + height - 0.25,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            x + width / 2,
            y + height / 2 - 0.15,
            body,
            ha="center",
            va="center",
            fontsize=10,
        )
    ax.annotate(
        "",
        xy=(6.85, 2.6),
        xytext=(3.15, 2.6),
        arrowprops=dict(arrowstyle="->", color="#52728b", linewidth=1.5),
    )
    ax.text(5, 2.75, r"$w\leftarrow w+R\,(v^+-v)$", ha="center", fontsize=12)
    ax.annotate(
        "",
        xy=(3.15, 1.95),
        xytext=(6.85, 1.95),
        arrowprops=dict(arrowstyle="->", color="#52728b", linewidth=1.5),
    )
    ax.text(5, 2.1, r"$v\leftarrow v+R^*(w^+-w)$", ha="center", fontsize=12)
    ax.text(
        5,
        1.2,
        "R : projection FE vers auxiliaire · R* : adjoint dans les masses",
        ha="center",
        fontsize=10,
    )
    r.save_figure(
        fig,
        "night-coupling",
        "Schéma des deux mises à jour. Les flèches correspondent à des sous-pas successifs. L'ancre FE supplémentaire conserve la relation entre les états quand le support et son opérateur de projection changent.",
    )
    r.figure(
        "night-fields",
        "Même domaine et mêmes échelles de couleur. Champ initial, référence, supports fixe et mobile, puis erreurs absolues. Le tracé est un échantillonnage FE; l'erreur L2 est intégrée séparément sur les triangles.",
    )
    r.section("Problème et protocole")
    r.paragraph(
        "Le domaine est le tore unité. Une vitesse cellulaire incompressible prescrite transporte jusqu'à t=0,25 une bosse de largeur 0,01, initialement centrée en (0,35; 0,45), de norme L2 proche de 1. L'espace FE principal possède 32768 triangles DG P4, soit 491520 coefficients par champ; la graduation fixe vaut 0,95. Le support principal auxiliaire comporte 64×64 cellules Q4, soit 102400 valeurs par champ. Deux supports orientés supplémentaires servent aux cisaillements diagonaux. Les méthodes n'ont donc pas le même nombre d'inconnues."
    )
    r.paragraph(
        "Le centre circulaire de l'énergie est calculé à partir de la solution, tous les quatre pas. Le support se déplace lorsque ce centre s'éloigne de plus de 0,03. Ce critère convient à une structure localisée; il n'est pas un estimateur d'erreur et ne traite pas plusieurs structures éloignées. La référence intègre les caractéristiques sur GPU par 512 pas RK4; les erreurs sont intégrées avec quadrature de degré 20 et subdivision indépendante. Les extrema sont échantillonnés."
    )
    r.figure(
        "night-trajectory",
        "Trajectoire du centre d'énergie et dissipation observée. Les croix indiquent les changements décidés à partir du champ numérique.",
    )
    r.section("Précision du support mobile")
    rows = [x["result"] for x in raw["moving-resolution"] if x["result"]["end"] == 0.25]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, adaptive in zip(axes, (False, True)):
        for n in (16, 32, 64):
            selected = sorted(
                [x for x in rows if x["adaptive"] == adaptive and x["naux"] == n],
                key=lambda x: x["steps"],
            )
            ax.loglog(
                [x["steps"] for x in selected],
                [max(x["l2"]) for x in selected],
                "o-",
                label=f"{n} × {n} Q4",
            )
        if rows:
            ax.axhline(
                rows[0]["best_fe_l2"][0],
                color="black",
                linestyle="--",
                label="Plancher FE",
            )
        ax.axhline(1e-3, color="#b74635", linestyle=":", label="Tolérance du benchmark")
        ax.set(
            title="Support mobile" if adaptive else "Support fixe",
            xlabel="Nombre de pas",
            ylabel="Erreur L2",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    r.save_figure(
        fig,
        "night-convergence",
        "À support fixe, réduire le pas ne corrige pas le manque de résolution le long de la trajectoire. Aucune pente temporelle universelle n'est revendiquée pour cette composition avec projections.",
    )
    r.section("Coût complet à précision imposée")
    r.paragraph(
        "Trois répétitions alternent DG eulérien et support mobile pour 1, 4 et 16 champs distincts et proches spatialement (graine 752). Chaque champ doit satisfaire L2 ≤ 0,001. Le coût inclut construction des opérateurs, détection initiale du centre, évolution, adaptations et transfert final. Le maillage FE et la projection initiale communs, ainsi que les diagnostics de référence, sont exclus des deux côtés. La compilation CUDA utilise le cache; ce ne sont pas des mesures de premier lancement sans cache."
    )
    summary = []
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, fields in zip(axes, (1, 4, 16)):
        statistics_rows = []
        for i, method in enumerate(("dg", "moving")):
            selected = [
                x
                for x in raw["matched"]
                if x["fields"] == fields and x["method"] == method
            ]
            if len(selected) != 3 or not all(x["accepted"] for x in selected):
                raise RuntimeError(
                    f"Incomplete or failed matched benchmark: {fields}, {method}"
                )
            totals = [x["result"]["total_seconds"] for x in selected]
            median = statistics.median(totals)
            preparation = statistics.median(
                x["result"]["setup_seconds"] for x in selected
            )
            finalization = statistics.median(
                x["result"].get("finalization_seconds", 0) for x in selected
            )
            ax.bar(
                i, preparation, color="#93afbd", label="Préparation" if i == 0 else None
            )
            ax.bar(
                i,
                median - preparation - finalization,
                bottom=preparation,
                color="#216b86",
                label="Évolution" if i == 0 else None,
            )
            ax.bar(
                i,
                finalization,
                bottom=median - finalization,
                color="#c99538",
                label="Retour FE" if i == 0 else None,
            )
            ax.errorbar(
                i,
                median,
                yerr=[[median - min(totals)], [max(totals) - median]],
                fmt="none",
                color="black",
                capsize=4,
            )
            statistics_rows.append(
                dict(
                    method=method,
                    median=median,
                    minimum=min(totals),
                    maximum=max(totals),
                    error=max(max(x["result"]["l2"]) for x in selected),
                )
            )
        speedup = statistics_rows[0]["median"] / statistics_rows[1]["median"]
        summary.append(dict(fields=fields, methods=statistics_rows, speedup=speedup))
        ax.set(
            xticks=[0, 1],
            xticklabels=["DG RK4", "Mobile"],
            ylabel="Temps total (s)",
            title=f"{fields} champ(s) · ratio {speedup:.1f}",
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    r.save_figure(
        fig,
        "night-cost",
        "Médiane et étendue des trois répétitions. Gain mesuré contre un DG amont RK4 à pas global sur ce maillage fortement gradué. Un DG à pas locaux et un solveur SL-DG GPU externe ne sont pas encore comparés.",
    )
    r.table(
        ["Champs", "DG (s)", "Mobile (s)", "Ratio", "L2 DG max", "L2 mobile max"],
        [
            [
                x["fields"],
                x["methods"][0]["median"],
                x["methods"][1]["median"],
                x["speedup"],
                x["methods"][0]["error"],
                x["methods"][1]["error"],
            ]
            for x in summary
        ],
    )
    r.paragraph(
        "Le pas DG retenu passe un contrôle de croissance sur champs aléatoires et la tolérance finale. Le réglage plus agressif (échelle 12) est rejeté : le contrôle aléatoire révèle une croissance d'énergie supérieure au seuil. Ce contrôle fini n'est pas un certificat spectral de stabilité. La variante mobile à 16 pas échoue à la tolérance; le benchmark utilise 64 pas."
    )
    krylov = [
        x["result"] for x in raw["krylov"] + raw["krylov-tuned"] + raw["krylov-fine"]
    ]
    if krylov:
        r.section("Comparateur DG à intégration exponentielle")
        r.paragraph(
            "Le même opérateur DG triangulaire est aussi intégré par projection d'Arnoldi de l'action exponentielle, avec flux central ou amont. Deux passages d'orthogonalisation et l'exponentielle de la petite matrice sont calculés sur GPU. Le mode constant est conservé explicitement. Cette construction classique, étudiée notamment par Saad (1992), fournit un comparateur à grands pas; la dimension de Krylov et le nombre de pas restent des paramètres de précision."
        )
        r.paragraph(
            "Le maillage, le champ initial et la tolérance L2 de 0,001 sont ceux du benchmark à un champ. La référence par caractéristiques et la quadrature indépendante sont construites une seule fois, hors chronométrage. Chaque coût inclut l'assemblage DG, un pas de chauffe et l'évolution; aucun graphe CUDA n'est employé pour Arnoldi. Le résidu final de Krylov est enregistré comme indicateur, tandis que l'acceptation repose sur l'erreur physique intégrée."
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for flux in ("central", "upwind"):
            for dimension in sorted({x["dimension"] for x in krylov}):
                selected = sorted(
                    [
                        x
                        for x in krylov
                        if x["flux"] == flux and x["dimension"] == dimension
                    ],
                    key=lambda x: x["steps"],
                )
                if selected:
                    label = f"{flux}, dimension {dimension}"
                    axes[0].loglog(
                        [x["steps"] for x in selected],
                        [max(x["l2"]) for x in selected],
                        "o-",
                        label=label,
                    )
                    axes[1].loglog(
                        [x["setup_seconds"] + x["evolution_seconds"] for x in selected],
                        [max(x["l2"]) for x in selected],
                        "o-",
                        label=label,
                    )
        for ax in axes:
            ax.axhline(0.001, color="black", linestyle="--", label="Tolérance")
            ax.axhline(
                max(krylov[0]["best_fe_l2"]),
                color="#888888",
                linestyle=":",
                label="Plancher FE",
            )
            ax.set_ylabel("Erreur L2")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        axes[0].set_xlabel("Nombre de pas exponentiels")
        axes[1].set_xlabel("Préparation et évolution (s)")
        r.save_figure(
            fig,
            "night-krylov",
            "Exploration des dimensions de Krylov et du nombre de pas. La stabilité de l'exponentielle projetée ne suffit pas à garantir sa précision; les configurations rejetées restent affichées.",
        )
        r.table(
            ["Flux", "Dimension", "Pas", "L2", "Accepté", "Temps (s)"],
            [
                [
                    x["flux"],
                    x["dimension"],
                    x["steps"],
                    max(x["l2"]),
                    "oui" if x["accepted"] else "non",
                    x["setup_seconds"] + x["evolution_seconds"],
                ]
                for x in krylov
            ],
        )
        accepted = [x for x in krylov if x["accepted"]]
        if accepted:
            fastest = min(
                accepted, key=lambda x: x["setup_seconds"] + x["evolution_seconds"]
            )
            r.paragraph(
                f"Le meilleur coût accepté dans cette exploration vaut {fastest['setup_seconds'] + fastest['evolution_seconds']:.4g} s, pour une erreur {max(fastest['l2']):.4g}. Une seule mesure par configuration est disponible ici; la sélection et les répétitions de chronométrage sont distinctes."
            )
        else:
            r.paragraph(
                "Aucune configuration exponentielle enregistrée n'atteint encore la tolérance. L'absence d'instabilité ne remplace pas la résolution suffisante de l'action exponentielle."
            )
    r.section("Mémoire : changement d'espace sans changer le champ FE")
    r.paragraph(
        "L'état est représenté par trois tableaux (v,w,a), avec v=R*w+(I−R*R)a. Son énergie est ||w||²+||a||²−||Ra||². La remise à zéro conserve v mais perd toute la mémoire invisible dans ce champ. Le transfert contractant conserve une partie de cette mémoire. En cas de raffinement exactement emboîté, l'injection corrige w par les nouveaux modes de a et préserve toute l'énergie augmentée."
    )
    memory = [x["result"] for x in raw["adaptive-memory"]]
    r.table(
        ["FE n", "Remise à zéro L2", "Mémoire L2", "Plancher FE"],
        [
            [
                n,
                next(
                    x["l2"][0]
                    for x in memory
                    if x["nfe"] == n and not x["carry_memory"]
                ),
                next(x["l2"][0] for x in memory if x["nfe"] == n and x["carry_memory"]),
                next(x["best_fe_l2"][0] for x in memory if x["nfe"] == n),
            ]
            for n in (64, 128)
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, n in zip(axes, (64, 128)):
        entries = sorted(
            [x for x in memory if x["nfe"] == n], key=lambda x: x["carry_memory"]
        )
        ax.bar([0, 1], [x["l2"][0] for x in entries], color=["#b76b51", "#216b86"])
        ax.axhline(
            entries[0]["best_fe_l2"][0],
            color="black",
            linestyle="--",
            label="Plancher FE",
        )
        ax.set(
            xticks=[0, 1],
            xticklabels=["Remise à zéro", "Transfert mémoire"],
            ylabel="Erreur L2",
            title=f"2 × {n}² triangles P4",
        )
        ax.legend()
    r.save_figure(
        fig,
        "night-memory",
        "Six changements de support, 128 pas et auxiliaire 64×64 Q4. Le transfert de mémoire améliore ici l'erreur finale; la contraction seule ne garantit pas cette amélioration pour toute EDP.",
    )
    projected = [x["result"] for x in raw["adaptive-projected"]]
    if projected:
        r.paragraph(
            "Une projection itérative supplémentaire utilise le recouvrement des deux espaces dans le même L2. Son filtre spectral contractant conserve davantage de mémoire lorsque le nombre d'itérations augmente. Cela ne garantit pas une diminution de l'erreur de transport. Les résultats ci-dessous sont rapportés séparément."
        )
        r.table(
            ["FE n", "Itérations", "L2", "Coût total (s)"],
            [
                [
                    x["nfe"],
                    x["projection_iterations"],
                    x["l2"][0],
                    x["setup_seconds"] + x["evolution_seconds"],
                ]
                for x in projected
            ],
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, n in zip(axes, (64, 128)):
            entries = sorted(
                [x for x in projected if x["nfe"] == n],
                key=lambda x: x["projection_iterations"],
            )
            ax.semilogx(
                [x["projection_iterations"] for x in entries],
                [x["l2"][0] / x["best_fe_l2"][0] for x in entries],
                "o-",
                base=2,
            )
            ax.axhline(1, color="black", linestyle="--")
            ax.set(
                title=f"FE : 2 × {n}² triangles",
                xlabel="Itérations de projection",
                ylabel="Erreur L2 / plancher FE",
            )
            ax.grid(alpha=0.2)
        r.save_figure(
            fig,
            "night-projected",
            "Conserver davantage de mémoire rapproche ici le résultat du plancher spatial. La projection itérative utilise le recouvrement physique des espaces, condition plus forte qu'un simple remap contractant.",
        )
    physics = [x["result"] for x in raw["adaptive-physics"]]
    if physics:
        r.section("Couplage avec une réaction FE spatiale")
        r.paragraph(
            "Une réaction de taux 1+0,5 sin(2πx) est appliquée par l'exponentielle exacte de son générateur Galerkin local, en deux demi-pas autour de chaque advection. Chaque mise à jour FE est réinjectée dans la mémoire. La référence indépendante intègre simultanément la trajectoire et l'exposant de réaction. L'énergie doit décroître; la masse n'est plus conservée par cette EDP."
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, n in zip(axes, (64, 128)):
            for label, carry, iterations in (
                ("Remise à zéro", False, 0),
                ("Transfert partiel", True, 0),
                ("Projection : 1", False, 1),
                ("Projection : 8", False, 8),
            ):
                entries = sorted(
                    [
                        x
                        for x in physics
                        if x["nfe"] == n
                        and x["carry_memory"] == carry
                        and x["projection_iterations"] == iterations
                    ],
                    key=lambda x: x["steps"],
                )
                ax.loglog(
                    [x["steps"] for x in entries],
                    [x["l2"][0] for x in entries],
                    "o-",
                    label=label,
                )
            entries = [x for x in physics if x["nfe"] == n]
            if entries:
                ax.axhline(
                    entries[0]["best_fe_l2"][0],
                    color="black",
                    linestyle="--",
                    label="Plancher FE",
                )
            ax.set(
                title=f"FE : 2 × {n}² triangles",
                xlabel="Nombre de pas",
                ylabel="Erreur L2",
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        r.save_figure(
            fig,
            "night-physics",
            "Transport et réaction calculés sur GPU. L'amélioration due à la projection de mémoire persiste dans ce couplage linéaire; cela ne valide pas encore une physique non linéaire ou un transport à vitesse FE générale.",
        )
    accelerated = [x["result"] for x in raw["adaptive-chebyshev"]]
    if accelerated:
        r.section("Accélération de la projection de mémoire")
        r.paragraph(
            "Le carré d'un résidu de Chebyshev fournit un filtre compris entre zéro et un sur tout le spectre admissible, même sous l'échelle de ciblage choisie. Il conserve donc la contraction et l'observation FE à chaque troncature. Les techniques polynomiales sont classiques; l'intérêt mesuré ici concerne leur insertion dans le transfert de mémoire. L'échelle 0,1 fonctionne mieux que 0,01 sur ce cas; aucune supériorité universelle du réglage n'est établie."
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, n in zip(axes, (64, 128)):
            selected = [x for x in projected if x["nfe"] == n]
            ax.plot(
                [x["projection_iterations"] - 1 for x in selected],
                [x["l2"][0] / x["best_fe_l2"][0] for x in selected],
                "o-",
                label="Itération simple",
            )
            for scale in (0.01, 0.1):
                selected = [
                    x
                    for x in accelerated
                    if x["nfe"] == n and x["filter_scale"] == scale
                ]
                ax.plot(
                    [2 * x["projection_iterations"] - 1 for x in selected],
                    [x["l2"][0] / x["best_fe_l2"][0] for x in selected],
                    "o-",
                    label=f"Chebyshev : s={scale}",
                )
            ax.axhline(1, color="black", linestyle="--")
            ax.set(
                title=f"FE : 2 × {n}² triangles",
                xlabel="Produits R R* par changement",
                ylabel="Erreur L2 / plancher FE",
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        r.save_figure(
            fig,
            "night-chebyshev",
            "Coût algébrique du filtre, hors transferts communs de préparation. Sept produits avec le filtre ciblé à 0,1 atteignent ici la précision de quinze produits avec l'itération simple. Les reconstructions géométriques restent dominantes dans le temps total.",
        )
    stress = [x["result"] for x in raw["memory-stress"]]
    if stress:
        r.paragraph(
            "Les trois transferts subissent chacun 2000 changements entre trois supports de graduations 0, 0,95 et 0,9999, entrelacés avec cisaillements et rotations locales de modes FE. Ces rotations sont un test algébrique isométrique, pas une réaction physique. Quatre champs aléatoires sont suivis; le champ FE est inchangé bit à bit au transfert."
        )
        r.table(
            ["Transfert", "Changements", "Défaut FE relatif max", "Défaut masse max"],
            [
                [
                    x["method"],
                    x["steps"],
                    x["maxima"]["observation_error"],
                    x["maxima"]["mass_error"],
                ]
                for x in stress
            ],
        )
    separated = [x["result"] for x in raw["moving-separated"]]
    if separated:
        r.section("Plusieurs structures : répartir les cellules selon l'énergie")
        r.paragraph(
            "Quatre champs distincts sont maintenant initialisés avec des centres espacés jusqu'à 0,15 autour de la position centrale. Le centre d'énergie unique ne suffit plus. Une alternative répartit les intervalles selon les marginales d'énergie dans x, y, x+y et x−y : histogrammes périodiques sur GPU, lissage gaussien de largeur 0,005, puissance 1/4, fond uniforme de 25 %, puis inversion de la masse cumulée. C'est une heuristique d'équidistribution; la stabilité vient des projections contractantes, pas d'une prétendue estimation d'erreur par ce moniteur."
        )
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        for ax, spread in zip(axes, (0.008, 0.05, 0.15)):
            for policy, label in (
                ("centroid", "Centre unique"),
                ("energy", "Équidistribution"),
            ):
                entries = sorted(
                    [
                        x
                        for x in separated
                        if x["spread"] == spread and x["mesh_policy"] == policy
                    ],
                    key=lambda x: x["naux"],
                )
                ax.loglog(
                    [x["naux"] for x in entries],
                    [max(x["l2"]) for x in entries],
                    "o-",
                    label=label,
                )
            entries = [x for x in separated if x["spread"] == spread]
            ax.axhline(
                max(entries[0]["best_fe_l2"]),
                color="black",
                linestyle="--",
                label="Plancher FE max",
            )
            ax.set(
                title=f"Dispersion : {spread}",
                xlabel="Cellules auxiliaires par axe",
                ylabel="Erreur L2 maximale",
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        r.save_figure(
            fig,
            "night-separated",
            "Dix-huit cas à quatre champs et 128 pas. Le maillage FE est fixé : son plancher se dégrade lorsque des profils passent dans ses régions moins fines. Le moniteur d'énergie rapproche l'erreur de ce plancher, sans résoudre cette limite FE.",
        )
        if (args.output / "night-separated-fields.pdf").exists():
            r.figure(
                "night-separated-fields",
                "Champs calculés pour une dispersion 0,15. Les champs affichés sont les sommes des quatre profils; les cartes d'erreur prennent le maximum des erreurs ponctuelles individuelles. Les valeurs L2 indiquées sont les maxima sur les quatre champs.",
            )
    coupled_separated = [x["result"] for x in raw["adaptive-separated"]]
    if coupled_separated:
        r.paragraph(
            f"Extension au couplage FE : {len(coupled_separated)} cas disponibles avec quatre structures séparées, avec ou sans réaction. Le tableau inclut le maillage FE raffiné pour lever le précédent plancher. Les erreurs sont prises sur le pire des quatre champs."
        )
        r.table(
            ["FE n", "Réaction", "Support", "Itér.", "L2 max", "Plancher max"],
            [
                [
                    x["nfe"],
                    "oui" if x["reaction"] else "non",
                    "énergie" if x["mesh_policy"] == "energy" else "centre",
                    x["projection_iterations"],
                    max(x["l2"]),
                    max(x["best_fe_l2"]),
                ]
                for x in coupled_separated
            ],
        )
    refined = [x["result"] for x in raw["separated-refinement"]]
    if refined:
        r.paragraph(
            "Le raffinement FE révèle une erreur auxiliaire auparavant masquée. L'étude suivante augmente séparément le nombre de cellules auxiliaires et le nombre de pas, sur le maillage FE n=256. Une amélioration de mémoire ne peut pas supprimer une erreur d'advection ou de résolution dans le support auxiliaire."
        )
        r.table(
            ["Aux. n", "Pas", "Réaction", "Itér.", "L2 max", "Plancher max"],
            [
                [
                    x["naux"],
                    x["steps"],
                    "oui" if x["reaction"] else "non",
                    x["projection_iterations"],
                    max(x["l2"]),
                    max(x["best_fe_l2"]),
                ]
                for x in refined
            ],
        )
    roundtrip = [x["result"] for x in raw["roundtrip-smoke"]]
    if roundtrip:
        r.section("Déformer puis revenir : erreur accumulée sans référence numérique")
        r.paragraph(
            "Le flot autonome est suivi pendant une durée T, puis sa vitesse est inversée pendant la même durée. La solution exacte finale est le profil initial : aucune intégration de caractéristiques n'est nécessaire pour cette référence. Les partitions restent adaptées au champ calculé pendant les deux trajets. La dissipation et les pertes de mémoire ne s'annulent pas lors du retour. Une erreur de phase réversible peut toutefois s'annuler : ce test complète les références aller simple sans les remplacer."
        )
        r.table(
            ["Durée aller", "Transfert", "Changements", "L2 max", "Retour FE max"],
            [
                [
                    x["half_duration"],
                    x["method"],
                    x["changes"],
                    max(x["l2"]),
                    max(x["fe_return_l2"]),
                ]
                for x in roundtrip
            ],
        )
    extended = [x["result"] for x in raw["roundtrip-long"]]
    if extended:
        r.paragraph(
            f"Campagne de déformation élargie : {len(extended)} cas sur 48 enregistrés. Les erreurs affichées portent sur le pire des quatre champs; les cas qui n'atteignent pas la tolérance restent présents."
        )
        r.paragraph(
            "Le contrôle de maillage de cette campagne a lieu tous les quatre pas numériques. Réduire le pas augmente donc aussi la fréquence physique de contrôle : ces courbes seules n'isolent pas l'erreur temporelle. Certains cas à 512 pas déclenchent presque un remaillage à chaque contrôle et deviennent plus coûteux et moins précis. L'ablation suivante teste séparément ce mécanisme."
        )
        for duration in (0.25, 1.0):
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            for ax, (nfe, naux) in zip(
                axes.ravel(), ((128, 64), (128, 128), (256, 64), (256, 128))
            ):
                entries = [
                    x
                    for x in extended
                    if x["nfe"] == nfe
                    and x["naux"] == naux
                    and x["half_duration"] == duration
                ]
                for method in ("reset", "projected", "chebyshev"):
                    selected = sorted(
                        [x for x in entries if x["method"] == method],
                        key=lambda x: x["steps_per_leg"],
                    )
                    if selected:
                        ax.loglog(
                            [x["steps_per_leg"] for x in selected],
                            [max(x["l2"]) for x in selected],
                            "o-",
                            label=method,
                        )
                if entries:
                    ax.axhline(
                        max(entries[0]["best_fe_l2"]),
                        color="black",
                        linestyle="--",
                        label="Plancher FE",
                    )
                    ax.legend(fontsize=8)
                ax.set(
                    title=f"FE n={nfe} · auxiliaire n={naux}",
                    xlabel="Pas par trajet",
                    ylabel="Erreur L2 maximale",
                )
                ax.grid(alpha=0.2)
            r.save_figure(
                fig,
                "night-return-" + str(duration).replace(".", "-"),
                f"Aller-retour de durée totale {2*duration}. Le raffinement en temps et en espace est distingué. Un panneau vide signifie que le calcul correspondant n'est pas encore enregistré.",
            )
    auxiliary_return = [x["result"] for x in raw["auxiliary-return"]]
    if auxiliary_return:
        r.paragraph(
            "Une ablation retire ensuite les observations FE intermédiaires et les transferts de mémoire. Le champ DG auxiliaire est transporté et remaillé directement, avec seulement une projection FE au départ et au retour. Elle isole les erreurs du transport auxiliaire et de son adaptation. Le nombre de cellules par direction vaut 128 ou 256; des grilles uniformes fixes servent de second contrôle. Cette version ne fournit pas de physique FE intermédiaire."
        )
        r.table(
            ["Aux. n", "Pas", "Intervalle", "Uniforme", "Changements", "L2 max"],
            [
                [
                    x["naux"],
                    x["steps_per_leg"],
                    x["check_every"],
                    "oui" if x["uniform"] else "non",
                    x["changes"],
                    max(x["l2"]),
                ]
                for x in auxiliary_return
            ],
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        for n in (128, 256):
            for uniform in (False, True):
                selected = [
                    x
                    for x in auxiliary_return
                    if x["naux"] == n and x["uniform"] == uniform
                ]
                positions = [
                    (
                        0
                        if x["steps_per_leg"] == 128
                        else (1 if x["check_every"] == 4 else 2)
                    )
                    for x in selected
                ]
                ax.semilogy(
                    positions,
                    [max(x["l2"]) for x in selected],
                    "o-",
                    label=f"Aux. {n}, "
                    + ("uniforme fixe" if uniform else "adaptation"),
                )
        memory = [
            x
            for x in extended
            if x["nfe"] == 256
            and x["naux"] == 128
            and x["half_duration"] == 1.0
            and x["method"] == "projected"
        ]
        if memory:
            ax.semilogy(
                [0 if x["steps_per_leg"] == 128 else 1 for x in memory],
                [max(x["l2"]) for x in memory],
                "s--",
                color="black",
                label="Couplage FE avec mémoire, aux. 128",
            )
        ax.set_xticks([0, 1, 2], ["128 / 4", "512 / 4", "512 / 16"])
        ax.set(
            xlabel="Pas par trajet / intervalle de contrôle",
            ylabel="Erreur L2 maximale au retour",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        r.save_figure(
            fig,
            "night-auxiliary-return",
            "Déformation et retour de durée totale deux. Les grilles adaptées sont décidées par leurs champs respectifs. Les points uniformes fixes apparaissent uniquement à 512 pas; le raffinement auxiliaire est distinct du raffinement temporel.",
        )
    stationary = [x["result"] for x in raw["monitor-stationary"]]
    if stationary:
        r.section("Diagnostic du moniteur : champ immobile et fréquence fixée")
        r.paragraph(
            "Un premier test supprime toute advection et reconstitue vingt-quatre fois le support à partir du même champ, avec projection entre supports. Il compare 5 et 12 points de Gauss par direction, ainsi qu'un plancher nul ou égal à un millionième de l'énergie marginale moyenne avant la racine quatrième. La surintégration conserve l'intégrale du carré du polynôme mais affine son histogramme spatial. Un déplacement résiduel dans ce test provient de la boucle moniteur-projection."
        )
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        for n, ax in zip((32, 64, 128), axes):
            for x in stationary:
                if x["naux"] != n:
                    continue
                ax.semilogy(
                    [h["iteration"] for h in x["history"]],
                    [max(h["displacement"], 1e-16) for h in x["history"]],
                    label=f"Gauss {x['sample_order']}, plancher {x['energy_floor']:g}",
                )
            ax.axhline(0.01, color="black", linestyle="--", label="Seuil de changement")
            ax.set(
                title=f"Auxiliaire n={n}",
                xlabel="Reconstruction sans advection",
                ylabel="Déplacement maximal des arêtes",
            )
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
        r.save_figure(
            fig,
            "night-monitor-stationary",
            "Test de la sensibilité du moniteur à sa représentation. Les projections peuvent dissiper le champ; il ne s'agit pas d'une itération à données ponctuelles fixes.",
        )
        initial_controls = [x for x in stationary if x["naux"] >= 64]
        if initial_controls:
            maximum = max(
                h["displacement"] for x in initial_controls for h in x["history"]
            )
            r.paragraph(
                f"Sur les supports n=64 et n=128, le déplacement maximal dans ce test immobile vaut {maximum:.4g}, à comparer au seuil 0,01 du contrôleur. La boucle appliquée au champ initial ne reproduit donc pas à elle seule les grands mouvements observés pendant l'évolution. Ce test ne suffit pas à attribuer leur cause à la quadrature."
            )
    monitor = [x["result"] for x in raw["monitor-return"]]
    if monitor:
        r.paragraph(
            "L'aller-retour utilise ensuite 128 pas avec contrôle tous les 4 pas, ou 512 pas avec contrôle tous les 4 ou 16 pas. Les réglages (128,4) et (512,16) ont le même intervalle physique entre contrôles. Le champ FE, le support auxiliaire et les huit itérations de transfert de mémoire sont identiques."
        )
        r.table(
            [
                "Pas",
                "Intervalle",
                "Gauss",
                "Plancher",
                "Changes",
                "L2 max",
                "Temps (s)",
            ],
            [
                [
                    x["steps_per_leg"],
                    x["check_every"],
                    x["sample_order"],
                    x["energy_floor"],
                    x["changes"],
                    max(x["l2"]),
                    x["setup_seconds"] + x["evolution_seconds"],
                ]
                for x in monitor
            ],
        )
    material = [
        x["result"]
        for x in raw["material-monitor"]
        + raw["material-phase"]
        + raw["material-minimax"]
    ]
    if material:
        r.section("La coupure périodique du moniteur déplace les quantiles")
        r.paragraph(
            "Des poids d'énergie positifs sont portés par des points matériels suivis par les caractéristiques. L'incompressibilité conserve ces poids. Ce diagnostic n'utilise aucune évolution FE/DG ni aucun remaillage du champ. Les grands déplacements des partitions sont reproduits à 512 et 1024 points par direction. Décaler la coupure périodique réduit le mouvement des quantiles diagonaux : une part importante du phénomène est géométrique, et non une instabilité de projection."
        )
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for axis, ax in enumerate(axes.ravel()):
            for x in material:
                label = (
                    (
                        "Phase minimax"
                        if x.get("phase_metric") == "maximum-displacement"
                        else "Phase probabiliste"
                    )
                    if x.get("floating_phase")
                    else ("Coupe décalée" if x["shifted"] else "Coupe fixe en zéro")
                )
                ax.plot(
                    [h["time"] for h in x["history"]],
                    [h["axis_displacement"][axis] for h in x["history"]],
                    "--" if x["samples"] == 512 else "-",
                    color={
                        "Coupe fixe en zéro": "#2166ac",
                        "Coupe décalée": "#d95f02",
                        "Phase probabiliste": "#7570b3",
                        "Phase minimax": "#1b9e77",
                    }[label],
                    label=f"{label}, {x['samples']}",
                )
            ax.axhline(0.01, color="black", linestyle=":")
            ax.set(
                title="Coordonnée " + ("x", "y", "x+y", "x−y")[axis],
                xlabel="Temps physique",
                ylabel="Déplacement maximal des arêtes",
            )
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=0.2)
        r.save_figure(
            fig,
            "night-periodic-cut",
            "Transport de poids matériels, sans transfert entre espaces. Les déplacements sont mesurés depuis le dernier changement accepté au seuil 0,01.",
        )
    fe_monitor = [x["result"] for x in raw["monitor-fe"]]
    if fe_monitor:
        r.paragraph(
            "Remplacer l'énergie auxiliaire par celle du champ FE observé laisse pratiquement inchangés le nombre de changements et l'erreur de retour. Le bruit propre à la représentation auxiliaire ne suffit donc pas à expliquer le phénomène."
        )
        r.table(
            ["Pas", "Intervalle", "Changements", "L2 max"],
            [
                [x["steps_per_leg"], x["check_every"], x["changes"], max(x["l2"])]
                for x in fe_monitor
            ],
        )
    phased = [x["result"] for x in raw["phase-return"]]
    if phased:
        r.paragraph(
            "Un alignement de phase par moindres carrés dans les coordonnées cumulées du moniteur supprime la dépendance à une coupure fixe. Il ne minimise toutefois pas le déplacement maximal des arêtes, qui déclenche ici les changements. Cette première variante ne réduit pas le nombre de remaillages ni l'erreur de retour; elle reste un résultat négatif du contrôleur."
        )
        r.table(
            ["Pas", "Intervalle", "Changements", "L2 max"],
            [
                [x["steps_per_leg"], x["check_every"], x["changes"], max(x["l2"])]
                for x in phased
            ],
        )
    minimax = [x["result"] for x in raw["minimax-return"]]
    if minimax:
        r.paragraph(
            "Une seconde phase minimise directement le plus grand déplacement physique des quantiles pour le nouveau moniteur fixé. La monotonie des quantiles permet de trouver par dichotomie le point où les déplacements extrêmes positifs et négatifs s'équilibrent. Cette optimalité locale ne minimise ni l'erreur du champ ni le nombre futur de changements."
        )
        r.table(
            ["Pas", "Intervalle", "Changements minimax", "L2 max", "Temps (s)"],
            [
                [
                    x["steps_per_leg"],
                    x["check_every"],
                    x["changes"],
                    max(x["l2"]),
                    x["evolution_seconds"],
                ]
                for x in minimax
            ],
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for rows, label in (
            (
                [
                    x["result"]
                    for x in raw["warm-return"]
                    if x["result"]["method"] == "projected"
                ],
                "Coupe fixe",
            ),
            (phased, "Phase probabiliste"),
            (minimax, "Phase minimax"),
        ):
            positions, counts, errors = [], [], []
            for i, (steps, interval) in enumerate(((128, 4), (512, 4), (512, 16))):
                selected = [
                    x
                    for x in rows
                    if x["steps_per_leg"] == steps and x["check_every"] == interval
                ]
                if selected:
                    positions.append(i)
                    counts.append(selected[0]["changes"])
                    errors.append(max(selected[0]["l2"]))
            axes[0].plot(positions, counts, "o-", label=label)
            axes[1].semilogy(positions, errors, "o-", label=label)
        for ax in axes:
            ax.set_xticks([0, 1, 2], ["128 / 4", "512 / 4", "512 / 16"])
            ax.set_xlabel("Pas par trajet / intervalle de contrôle")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Nombre de changements")
        axes[1].set_ylabel("Erreur L2 maximale au retour")
        r.save_figure(
            fig,
            "night-phase-comparison",
            "Même transfert de mémoire tronqué, trois choix de coupure périodique. La troisième configuration conserve l'intervalle physique de contrôle de la première.",
        )
    warm_validation = args.input / "warm-validation.json"
    if warm_validation.exists():
        controls = json.loads(warm_validation.read_text())["results"]
        r.section("Un transfert sans changement doit conserver la mémoire")
        r.paragraph(
            "Le filtre polynomial initial démarre de zéro. Même quand l'ancien et le nouveau support coïncident, il multiplie chaque mode de mémoire par un facteur inférieur à un : un nombre fini d'itérations introduit donc un défaut pour une opération géométriquement nulle. Ce défaut est distinct de la stabilité énergétique. Le contrôle ci-dessous utilise des états aléatoires après transport et physique, et calcule leur distance dans un espace ambiant commun."
        )
        r.table(
            ["Degré", "Transfert", "Distance au carré", "Perte d'énergie"],
            [
                [
                    x["degree"],
                    z["method"],
                    max(z["ambient_change_squared"]),
                    max(z["energy_loss"]),
                ]
                for x in controls
                for z in x["records"]
                if z["motion"] == 0
            ],
        )
        r.paragraph(
            "Une variante initialise l'itération avec la représentation auxiliaire précédente transférée, puis borne radialement son énergie invisible par celle de l'ancien état. Elle conserve le champ FE et son énergie augmentée, et devient l'identité ambiante à support inchangé. Le facteur dépend de l'état : cette variante est non linéaire, et aucune contraction de la distance entre deux solutions n'est déduite de son seul bilan d'énergie. La correction d'un défaut d'identité ne garantit pas une meilleure erreur physique."
        )
        counterexample = json.loads(warm_validation.read_text()).get(
            "limiter_counterexample"
        )
        if counterexample:
            r.paragraph(
                f"Un contre-exemple GPU dans un espace ambiant de dimension trois confirme cette limite : le rapport de distance entre deux états vaut {counterexample['pair_distance_gain']:.6g}, alors que l'énergie de chaque état reste bornée. La variante n'est donc pas non expansive."
            )
        warm = [x["result"] for x in raw["warm-return"]]
        if warm:
            r.table(
                [
                    "Pas",
                    "Intervalle",
                    "Transfert",
                    "Changements",
                    "L2 max",
                    "Temps (s)",
                ],
                [
                    [
                        x["steps_per_leg"],
                        x["check_every"],
                        x["method"],
                        x["changes"],
                        max(x["l2"]),
                        x["evolution_seconds"],
                    ]
                    for x in warm
                ],
            )
    galerkin_file = args.input / "galerkin.validation.json"
    if galerkin_file.exists():
        controls = json.loads(galerkin_file.read_text())["results"]
        r.section("Projection de mémoire dans un petit espace augmenté")
        r.paragraph(
            "L'état précédent transféré est ajouté à un petit espace de Krylov. La projection orthogonale de la mémoire sur son image orthogonale au champ FE se calcule avec une petite matrice de Gram GPU. Elle conserve le champ FE et vérifie une identité de Pythagore : perte d'énergie égale carré du déplacement ambiant. L'espace contient la mémoire entrante lorsque le support est inchangé. Un seuil de rang 10⁻¹² limite ces propriétés en arithmétique flottante. L'augmentation de Krylov est classique; cette réalisation est évaluée ici dans le transfert FE."
        )
        r.table(
            [
                "Degré",
                "Défaut observation max",
                "Défaut oracle max",
                "Défaut Pythagore max",
            ],
            [
                [
                    x["degree"],
                    max(z["observation_error"] for z in x["records"]),
                    max(z["oracle_error"] for z in x["records"]),
                    max(
                        abs(a - b)
                        for z in x["records"]
                        for a, b in zip(z["energy_loss"], z["ambient_change_squared"])
                    ),
                ]
                for x in controls
            ],
        )
        tested = [x["result"] for x in raw["galerkin"]]
        if tested:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            transfer_cases = [x["result"] for x in raw["warm-return"]] + tested
            for method, label in (
                ("projected", "Filtre tronqué"),
                ("warm", "Initialisation chaude"),
                ("galerkin", "Projection augmentée"),
            ):
                selected = [x for x in transfer_cases if x["method"] == method]
                positions, errors, costs = [], [], []
                for i, (steps, interval) in enumerate(((128, 4), (512, 4), (512, 16))):
                    group = [
                        x
                        for x in selected
                        if x["steps_per_leg"] == steps and x["check_every"] == interval
                    ]
                    if group:
                        positions.append(i)
                        errors.append(max(group[0]["l2"]))
                        costs.append(group[0]["evolution_seconds"])
                axes[0].semilogy(positions, errors, "o-", label=label)
                axes[1].plot(positions, costs, "o-", label=label)
            for ax in axes:
                ax.set_xticks([0, 1, 2], ["128 / 4", "512 / 4", "512 / 16"])
                ax.set_xlabel("Pas par trajet / intervalle de contrôle")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
            axes[0].set_ylabel("Erreur L2 maximale au retour")
            axes[1].set_ylabel("Temps d'évolution (s)")
            r.save_figure(
                fig,
                "night-galerkin",
                "Trois réalisations du transfert de mémoire avec le même contrôleur de maillage. Les trajectoires de supports peuvent différer parce que le contrôleur dépend du champ calculé. Une exécution par configuration.",
            )
        r.table(
            ["Pas", "Intervalle", "Changements", "L2 max", "Temps (s)"],
            [
                [
                    x["steps_per_leg"],
                    x["check_every"],
                    x["changes"],
                    max(x["l2"]),
                    x["evolution_seconds"],
                ]
                for x in tested
            ],
        )
        r.paragraph(
            "Le sous-espace est construit à partir de chaque champ : le transfert est non linéaire. L'identité de Pythagore ne démontre pas une contraction de la distance entre deux solutions, ni un gain d'erreur pour l'EDP. Les tests de retour servent à examiner ce second point."
        )
    stability_file = args.input / "galerkin-stability.json"
    if stability_file.exists():
        stability = json.loads(stability_file.read_text())["results"]
        r.paragraph(
            "Un contre-exemple indépendant en dimension six précise la limite de stabilité. Deux entrées séparées de 1e-7 sont projetées sur des espaces augmentés qui dépendent de l'état. Les sorties peuvent être beaucoup plus éloignées, alors que chaque énergie décroît et que chaque identité de Pythagore est satisfaite. La note donne la construction explicite et les gains limites analytiques. Cette observation exclut une preuve de non-expansivité de ce transfert non linéaire."
        )
        r.table(
            ["Directions", "Rang effectif", "Gain de distance", "Limite analytique"],
            [
                [
                    x["directions"],
                    x["effective_directions"],
                    f"{x['distance_gain']:.5g}",
                    x["limiting_gain"],
                ]
                for x in stability
            ],
        )
    frozen = [x["result"] for x in raw["frozen-transfer"]]
    if frozen:
        r.section("Comparaison des transferts à géométrie identique")
        r.paragraph(
            "Des points matériels transportent les poids d'énergie de la solution initiale et définissent à l'avance les partitions. Les huit méthodes suivent exactement cette même succession de 63 changements, puis les mêmes opérateurs auxiliaires. Les matrices FE–auxiliaire et les projections entre supports sont construites une seule fois et partagées. La rétroaction du champ calculé sur le maillage est ainsi supprimée de cette comparaison."
        )
        r.paragraph(
            "À chaque changement, trois mesures partent du même état, après une application de chauffe. Leur médiane inclut le transfert et ses diagnostics. La somme de ces médianes isole le coût des transferts; elle exclut la construction géométrique commune et n'est pas un temps complet de résolution. Les états des méthodes peuvent différer parce que leurs erreurs se sont accumulées."
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        labels = [
            x["method"]
            .replace("galerkin0", "Galerkin 1 dir.")
            .replace("galerkin1", "Galerkin 2 dir.")
            .replace("galerkin2", "Galerkin 3 dir.")
            .replace("galerkin4", "Galerkin 5 dir.")
            for x in frozen
        ]
        axes[0].bar(range(len(frozen)), [max(x["l2"]) for x in frozen])
        axes[0].set_yscale("log")
        axes[0].axhline(
            max(frozen[0]["best_fe_l2"]),
            color="black",
            linestyle="--",
            label="Plancher FE",
        )
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel("Erreur L2 maximale au retour")
        axes[1].bar(range(len(frozen)), [x["transfer_seconds"] for x in frozen])
        axes[1].set_ylabel("Somme des temps de transfert (s)")
        for ax in axes:
            ax.set_xticks(range(len(frozen)), labels, rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.2)
        r.save_figure(
            fig,
            "night-frozen-transfer",
            "Même suite de supports et mêmes transports. Les variantes de Galerkin utilisent une, deux, trois ou cinq directions; la matrice de Gram peut avoir un rang numérique plus faible. La seconde colonne mesure uniquement les transferts avec diagnostics.",
        )
        r.paragraph(
            f"Construction initiale commune : {frozen[0]['common_setup_seconds']:.3g} s; constructions géométriques communes pendant l'évolution : {frozen[0]['common_rebuild_seconds']:.3g} s. Ces coûts restent à payer pour une exécution indépendante."
        )
    block_file = args.input / "block-validation.json"
    if block_file.exists():
        block_controls = [
            item
            for group in json.loads(block_file.read_text())["results"]
            for item in group["records"]
        ]
        r.section("Une mémoire commune aux composantes du système")
        r.paragraph(
            "Les espaces de transfert construits séparément pour chaque champ dépendent du choix de base des composantes. Une variante réunit leurs directions dans un espace partagé et projette tous les champs sur cet espace. Elle préserve le champ FE et l'identité au support inchangé. La perte de la matrice complète des produits scalaires entre champs est positive semi-définie; une rotation orthogonale des composantes produit la même rotation de la sortie. Les preuves, à géométrie et métrique des composantes fixées, sont données dans la note."
        )
        r.table(
            ["Contrôle GPU, degrés 1, 2 et 4", "Défaut maximal"],
            [
                [label, max(x[key] for x in block_controls)]
                for label, key in (
                    ("Observation FE", "observation_error"),
                    ("Pythagore, matrice complète", "pythagoras_error"),
                    ("Rotation des champs, espace partagé", "covariance_error"),
                    (
                        "Rotation des champs, espaces séparés",
                        "individual_covariance_error",
                    ),
                )
            ],
        )
        r.paragraph(
            "Le défaut de covariance est ici le maximum absolu des différences de coefficients de l'ancre et du champ auxiliaire après rotation. Les champs de contrôle sont aléatoires; ce résultat teste une propriété algébrique et ne mesure pas une erreur d'EDP. L'espace partagé reste dépendant de l'ensemble des états et ne supprime pas le contre-exemple de stabilité entre entrées distinctes."
        )
    block_frozen = [x["result"] for x in raw["block-frozen"]]
    if block_frozen:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = [x["method"] for x in block_frozen]
        axes[0].bar(labels, [max(x["l2"]) for x in block_frozen])
        axes[0].set(yscale="log", ylabel="Erreur L2 maximale")
        axes[1].bar(labels, [x["transfer_seconds"] for x in block_frozen])
        axes[1].set(ylabel="Somme des temps de transfert (s)")
        for ax in axes:
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.2)
        r.save_figure(
            fig,
            "night-block-transfer",
            "Tous les champs et toutes les méthodes suivent la même suite de 63 changements de support. Les constructions géométriques sont communes; les temps de transfert incluent les diagnostics. Les variantes block réunissent les directions des quatre champs.",
        )
        r.paragraph(
            f"La géométrie commune coûte {block_frozen[0]['common_setup_seconds']:.3g} s initialement, puis {block_frozen[0]['common_rebuild_seconds']:.3g} s pendant le trajet. Ce coût doit être ajouté pour une résolution indépendante."
        )
    reduction_file = args.input / "reduction-validation.json"
    if reduction_file.exists() and "timings" in json.loads(reduction_file.read_text()):
        reductions = json.loads(reduction_file.read_text())
        r.section("Réduction CUDA des produits de Gram")
        r.paragraph(
            "Le profilage synchronisé identifie la formation des petits produits de Gram comme le surcoût de Galerkin : les contractions emploient une très longue dimension spatiale et peu de colonnes. Une réduction CUDA divise cette somme en blocs de 1024 degrés de liberté, puis réunit les sommes partielles sur GPU. La diagonalisation et le critère de rang restent identiques. Cinq répétitions entrelacées comparent les deux assemblages sur le même état; ces temps concernent un transfert et incluent ses diagnostics."
        )
        r.table(
            [
                "Directions/champ",
                "Espace",
                "Initial (ms)",
                "Réduction (ms)",
                "Écart ambiant",
            ],
            [
                [
                    x["iterations"] + 1,
                    "partagé" if x["shared"] else "séparé",
                    1e3 * x["median_seconds"]["einsum"],
                    1e3 * x["median_seconds"]["reduction"],
                    x["ambient_error"],
                ]
                for x in reductions["timings"]
            ],
        )
        r.paragraph(
            "Les contrôles indépendants comparent les produits pondérés, les rangs retenus, les coefficients et l'état ambiant. Les directions presque nulles peuvent amplifier les différences de coefficients; le fichier brut conserve également la variabilité entre répétitions du calcul initial. Le gain local doit être distingué du temps de construction des supports."
        )
    tiled_file = args.input / "carrier-tiled.json"
    if tiled_file.exists():
        tiled = json.loads(tiled_file.read_text())
        r.section("Évaluer la quadrature en parallèle pour reconstruire les supports")
        r.paragraph(
            "Le noyau précédent évaluait chaque point de quadrature avec une seule voie du warp avant de répartir les entrées de la matrice. Le nouveau noyau prépare 32 points en parallèle dans la mémoire partagée, puis accumule les mêmes produits de bases. La normalisation par les poids de masse est effectuée à la fin de l'intégration. Les intersections, quadratures et espaces polynomiaux restent identiques. L'option tiled est explicite; les campagnes précédentes utilisaient l'ancien constructeur."
        )
        checks = tiled["controls"]
        r.paragraph(
            f"Les {len(checks)} contrôles comparent le constructeur au noyau scalaire indépendant : degrés 1, 2, 4, partitions uniformes ou périodiques non uniformes, intégrales ordinaires ou pondérées par la réaction, et deux subdivisions de quadrature. L'écart relatif maximal des blocs est {max(x['relative_block_error'] for x in checks):.3g}. Les applications et l'adjoint de masse sont également comparés."
        )
        timings = tiled.get("timings", [])
        if timings:
            r.table(
                [
                    "FE n",
                    "Aux. n",
                    "Ancien (s)",
                    "Quadrature parallèle (s)",
                    "Rapport",
                    "Écart L2 après 16 transports",
                ],
                [
                    [
                        x["nfe"],
                        x["naux"],
                        x["median_total_seconds"]["warp"],
                        x["median_total_seconds"]["tiled"],
                        x["median_total_seconds"]["warp"]
                        / x["median_total_seconds"]["tiled"],
                        x["transport_l2"],
                    ]
                    for x in timings
                ],
            )
            fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
            for offset, builder, label in (
                (-0.18, "warp", "Construction précédente"),
                (0.18, "tiled", "Quadrature parallèle"),
            ):
                ax.bar(
                    [i + offset for i in range(len(timings))],
                    [x["median_total_seconds"][builder] for x in timings],
                    width=0.34,
                    label=label,
                )
            ax.set_xticks(
                range(len(timings)),
                [f"FE {x['nfe']}, auxiliaire {x['naux']}" for x in timings],
            )
            ax.set(ylabel="Construction complète (s)", yscale="log")
            ax.legend()
            ax.grid(axis="y", alpha=0.2)
            r.save_figure(
                fig,
                "night-carrier-tiled",
                "Médianes de cinq constructions après chauffe, dans un ordre alterné. La construction géométrique, la compaction des blocs et les allocations sont incluses. Les temps ne sont pas ceux d'une résolution complète.",
            )
        r.paragraph(
            "Compute Sanitizer memcheck ne signale aucune erreur sur la validation entière. Le contrôle racecheck ciblant le nouveau noyau ne signale ni erreur ni avertissement. Le contrôle non filtré émet 48 avertissements dans le noyau de scan de CuPy utilisé par nonzero, et aucun dans le nouveau noyau. Les journaux non filtrés et ciblés sont conservés; ces avertissements de bibliothèque ne sont pas présentés comme résolus."
        )
    repeated = [x["result"] for x in raw["long-return-benchmark"]]
    pilot = [x["result"] for x in raw["tiled-return-pilot"]]
    if repeated or pilot:
        r.section("Coût complet après accélération de la reconstruction géométrique")
        r.paragraph(
            "Le retour long utilise FE n = 256, auxiliaire n = 128, quatre champs, deux trajets de durée 1 et 512 pas par trajet. Le coût inclut la préparation du couplage, les contrôles, les changements géométriques et l'évolution. La quadrature d'erreur finale reste hors chronométrage, comme dans la campagne initiale. Les répétitions alternent l'ordre des configurations."
        )
        if repeated:
            groups = sorted({(x["carrier_builder"], x["method"]) for x in repeated})
            rows = []
            for builder, method in groups:
                xs = [
                    x
                    for x in repeated
                    if (x["carrier_builder"], x["method"]) == (builder, method)
                ]
                seconds = [x["complete_seconds"] for x in xs]
                errors = [max(x["l2"]) for x in xs]
                rows.append(
                    [
                        builder,
                        method,
                        len(xs),
                        statistics.median(seconds),
                        min(seconds),
                        max(seconds),
                        min(errors),
                        max(errors),
                    ]
                )
            r.table(
                [
                    "Construction",
                    "Transfert",
                    "N",
                    "Médiane (s)",
                    "Min. (s)",
                    "Max. (s)",
                    "L2 min.",
                    "L2 max.",
                ],
                rows,
            )
        else:
            x = pilot[0]
            r.paragraph(
                f"Un premier calcul avec Galerkin à trois directions coûte {x['complete_seconds']:.3g} s et donne une erreur maximale L2 de {max(x['l2']):.5g}, avec {x['changes']} changements de support. L'exécution précédente utilisait environ 255 s avec la même méthode et l'ancien constructeur. Cette comparaison initiale est une seule exécution; les répétitions sont rapportées dès qu'elles sont terminées."
            )
        r.paragraph(
            "Les réductions atomiques et les petits Grams proches d'un défaut de rang peuvent amplifier les différences d'arrondi dans le calcul adaptatif. Les intervalles d'erreur des répétitions sont donc conservés à côté des temps; un accord à l'arrondi des blocs n'implique pas un accord à l'arrondi du champ après 255 changements non linéaires."
        )
    fast_return = [x["result"] for x in raw["fast-return"]]
    if fast_return:
        r.section("Coût du trajet complet avec la réduction CUDA")
        r.paragraph(
            "Le changement d'assemblage est ensuite évalué dans le contrôleur adaptatif complet. Les coûts incluent la préparation, les contrôles et toutes les reconstructions géométriques. Les espaces partagés sont comparés aux projections séparées; la métrique d'erreur reste le maximum des quatre erreurs L2 après retour exact."
        )
        r.table(
            [
                "FE / auxiliaire",
                "Durée/branche",
                "Pas / contrôle",
                "Mémoire",
                "Changements",
                "L2 max",
                "Coût (s)",
            ],
            [
                [
                    f"{x['nfe']} / {x['naux']}",
                    x["half_duration"],
                    f"{x['steps_per_leg']} / {x['check_every']}",
                    x["method"],
                    x["changes"],
                    max(x["l2"]),
                    x["setup_seconds"] + x["evolution_seconds"],
                ]
                for x in fast_return
            ],
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, duration, nfe, naux in zip(axes, (0.25, 1.0), (128, 256), (64, 128)):
            groups = (
                (
                    [
                        x["result"]
                        for x in raw["roundtrip-long"]
                        if x["result"]["method"] == "chebyshev"
                    ],
                    "Chebyshev",
                ),
                (
                    [
                        x["result"]
                        for x in raw["galerkin"] + raw["joint-return"]
                        if not x["result"].get("floating_phase", False)
                    ],
                    "Galerkin initial",
                ),
                (
                    [x for x in fast_return if x["method"] == "galerkin"],
                    "Galerkin, réduction",
                ),
                (
                    [x for x in fast_return if x["method"] == "block"],
                    "Espace partagé, réduction",
                ),
            )
            for values, label in groups:
                selected = [
                    x
                    for x in values
                    if x["half_duration"] == duration
                    and x["nfe"] == nfe
                    and x["naux"] == naux
                ]
                if selected:
                    ax.scatter(
                        [x["setup_seconds"] + x["evolution_seconds"] for x in selected],
                        [max(x["l2"]) for x in selected],
                        label=label,
                    )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="Préparation et évolution (s)",
                ylabel="Erreur L2 maximale",
                title=f"Durée {duration} par branche",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        r.save_figure(
            fig,
            "night-fast-return",
            "Le gain sur les petits produits de Gram se répercute partiellement sur le coût complet, dominé par les reconstructions des supports. Les points comprennent plusieurs pas et fréquences de contrôle; le tableau conserve leurs paramètres.",
        )
    systems = [x["result"] for x in raw["system-coupling"]]
    if systems:
        r.section("Réaction FE à quatre composantes et changement de base")
        r.paragraph(
            "Le système transporte quatre profils et applique la réaction spatiale diagonale de taux (0,5; 1; 2; 4) fois 1+0,5 sin(2πx). Le même problème est résolu après une rotation orthogonale des composantes, où la réaction devient couplée. Les résultats sont ramenés dans la base initiale. Une suite matérielle de supports est imposée à toutes les variantes pour isoler cette propriété du contrôleur. Les demi-pas de réaction sont effectués dans l'espace FE; le transport et la mémoire auxiliaire restent sur GPU."
        )
        r.table(
            [
                "FE",
                "Dispersion",
                "Méthode",
                "L2, base initiale",
                "L2, base tournée",
                "Écart entre bases",
            ],
            [
                [
                    x["nfe"],
                    x["spread"],
                    x["method"],
                    max(x["diagonal_basis_errors"]["l2"]),
                    max(x["rotated_basis_errors"]["l2"]),
                    x["fe_covariance_l2"],
                ]
                for x in systems
            ],
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, spread in zip(axes, (0.008, 0.15)):
            for method in ("chebyshev", "galerkin2", "block2"):
                selected = sorted(
                    [
                        x
                        for x in systems
                        if x["spread"] == spread and x["method"] == method
                    ],
                    key=lambda x: x["nfe"],
                )
                if selected:
                    ax.semilogy(
                        [x["nfe"] for x in selected],
                        [max(x["fe_covariance_l2"], 1e-16) for x in selected],
                        "o-",
                        label=method,
                    )
            ax.set(
                xlabel="Résolution FE",
                ylabel="Norme L2 de la différence entre bases",
                title=f"Dispersion initiale {spread}",
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        r.save_figure(
            fig,
            "night-system-covariance",
            "Le changement de coordonnées des composantes décrit la même équation physique. L'écart affiché est une différence entre deux solutions FE, distincte de leur erreur par rapport à la référence caractéristique, donnée dans le tableau.",
        )
    mixtures = [x["result"] for x in raw["mixture"]]
    if mixtures:
        r.section("Fermeture locale de quatre fractions de mélange")
        r.paragraph(
            "Quatre fractions initialement positives ont une somme exactement égale à un. Elles sont transportées, avec ou sans réaction spatiale d'échange entre espèces définie par le laplacien du graphe cyclique à quatre sommets. Cette réaction conserve la somme. Une projection commune à tous les champs préserve toute relation linéaire dont le second membre appartient à l'espace FE; elle doit donc conserver cette fermeture jusque dans la mémoire auxiliaire. La positivité des fractions reste un contrôle séparé."
        )
        r.table(
            [
                "FE / auxiliaire",
                "Réaction",
                "Méthode",
                "L2 max",
                "Fermeture FE",
                "Fermeture auxiliaire",
                "Masse totale",
            ],
            [
                [
                    f"{x['nfe']} / {x['naux']}",
                    "oui" if x["reactive"] else "non",
                    x["method"],
                    max(x["l2"]),
                    x["fe_closure_l2"],
                    x["auxiliary_closure_l2"],
                    x["total_mass_error"],
                ]
                for x in mixtures
            ],
        )
        for reactive in (False, True):
            name = f"night-mixture-closure-128-{int(reactive)}"
            if (args.output / (name + ".pdf")).exists():
                caption = (
                    "Défaut de fermeture local, "
                    + ("avec réaction d'échange" if reactive else "sans réaction")
                    + ". Chaque couleur donne la moyenne quadratique exacte par cellule FE de la somme des fractions moins un, en incluant tous les coefficients P4. La même échelle logarithmique est appliquée aux trois méthodes."
                )
                r.figure(name, caption)
        r.paragraph(
            "La cohérence des fractions de mélange est une exigence déjà traitée, notamment par Plewa et Müller dans la méthode CMA. L'expérience examine ici sa conservation dans un couplage FE à mémoire cachée et supports variables; elle ne revendique pas l'invention de cette contrainte."
        )
    relaxation = [x["result"] for x in raw["relaxation"]]
    if relaxation:
        r.section("Une réaction coercive doit amortir la mémoire cachée")
        r.paragraph(
            "Une réaction scalaire uniforme de taux κ commute avec l'advection et multiplie toute la solution par exp(−κT). Le couplage FE initial applique la réaction à l'observation et conserve la composante orthogonale. Il est contractant, mais ne reproduit pas nécessairement cette décroissance physique. Une extension impose q nouveau = G(v) + γ(q−v), avec γ = exp(−κΔt) pour un sous-pas de réaction uniforme. Son énergie est exactement la somme de l'énergie FE après physique et de γ² fois l'énergie cachée initiale."
        )
        r.paragraph(
            "Le contrôle conserve un support auxiliaire uniforme fixe et utilise le même transport numérique sans réaction comme référence de commutation. Les erreurs physiques sont aussi comparées à une référence caractéristique indépendante. Les erreurs divisées par exp(−κT) exposent les écarts que masquerait une solution presque nulle. Cette extension est exacte pour une réaction uniforme; une physique générale requiert un modèle spécifique pour les sous-échelles."
        )
        selected = [x for x in relaxation if x["nfe"] == 64 and x["steps"] == 256]
        r.table(
            [
                "Taux κ",
                "Amortissement",
                "Écart de commutation / exp(−κT)",
                "Erreur L2 physique",
            ],
            [
                [
                    x["strength"],
                    "FE" if x["method"] == "visible-only" else "FE et mémoire",
                    x["normalized_commutation_l2"],
                    max(x["physical_l2"]),
                ]
                for x in selected
            ],
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, nfe in zip(axes, (32, 64)):
            for method in ("visible-only", "visible-and-hidden"):
                for strength in (1.0, 10.0, 100.0):
                    rows = sorted(
                        [
                            x
                            for x in relaxation
                            if x["nfe"] == nfe
                            and x["method"] == method
                            and x["strength"] == strength
                        ],
                        key=lambda x: x["steps"],
                    )
                    label = (
                        "FE" if method == "visible-only" else "FE et mémoire"
                    ) + f", κ={strength:g}"
                    ax.loglog(
                        [x["steps"] for x in rows],
                        [max(x["normalized_errors"]["l2"]) for x in rows],
                        "o-" if method == "visible-only" else "s--",
                        label=label,
                    )
            ax.set(
                xlabel="Nombre de pas",
                ylabel="Erreur L2 / exp(−κT)",
                title=f"Résolution FE {nfe}",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=6)
        r.save_figure(
            fig,
            "night-relaxation",
            "Avec amortissement de la mémoire, les trois intensités de réaction redonnent l'erreur du même transport sans réaction, après normalisation. Les courbes correspondantes se superposent. La conservation d'énergie cachée par le couplage FE seul ne décrit pas la physique d'une relaxation uniforme forte.",
        )
    moments = [
        x["result"]
        for name in ("reaction-moments", "reaction-moments-fine")
        for x in raw[name]
    ]
    if moments:
        r.section("Réagir sur le champ complet, puis en conserver les moments")
        r.paragraph(
            "Pour la réaction de taux κ(1 + 0,5 sin(2πx)), amortir uniformément la mémoire à la vitesse minimale assure une borne d'énergie, mais ne décrit pas son évolution spatiale. La nouvelle opération calcule les moments FE et auxiliaires de gq, où g est l'exponentielle physique de la réaction et q le champ complet. Trois blocs de masse pondérés suffisent : FE–FE, FE–auxiliaire et auxiliaire–auxiliaire. La projection dans trois directions cachées conserve exactement les moments FE de gq et possède une identité de Pythagore, sous les hypothèses de quadrature et de rang de la note théorique."
        )
        r.paragraph(
            "Le test conserve le même transport cellulaire, le même maillage FE P4 gradué et un support auxiliaire uniforme fixe, deux fois plus fin dans chaque direction. Le temps final est 0,25. La référence par caractéristiques est indépendante; l'écart entre 512 et 1024 sous-pas de référence est inférieur à 3 × 10⁻¹¹ relativement au maximum de la référence. Les erreurs sont divisées par sa norme L2 finale : pour κ = 100 cette norme vaut seulement 4,20 × 10⁻¹⁴. Les écarts relatifs très élevés des modèles incomplets décrivent donc des résidus faibles en valeur absolue, mais très supérieurs au champ physique attendu."
        )
        methods = ("fe-only", "coercive-lift", "ambient-moments")
        labels = dict(
            zip(methods, ("Réaction FE", "Mémoire amortie", "Moments complets"))
        )
        selected = [x for x in moments if x["steps"] == 256 and x["strength"] == 100]
        r.table(
            [
                "FE n",
                "Modèle",
                "L2 physique",
                "L2 relative",
                "Plancher relatif",
                "Évolution (s)",
            ],
            [
                [
                    x["nfe"],
                    labels[x["method"]],
                    x["l2"][0],
                    x["relative_l2"],
                    x["relative_fe_floor"],
                    x["evolution_seconds"],
                ]
                for x in selected
            ],
        )
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
        for ax, nfe in zip(axes, sorted({x["nfe"] for x in moments})):
            for method in methods:
                rows = sorted(
                    [
                        x
                        for x in moments
                        if x["nfe"] == nfe
                        and x["steps"] == 256
                        and x["method"] == method
                    ],
                    key=lambda x: x["strength"],
                )
                ax.loglog(
                    [x["strength"] for x in rows],
                    [x["relative_l2"] for x in rows],
                    "o-",
                    label=labels[method],
                )
            ax.loglog(
                [x["strength"] for x in rows],
                [x["relative_fe_floor"] for x in rows],
                "k--",
                label="Plancher FE",
            )
            ax.set(
                xlabel="Intensité κ",
                ylabel="Erreur L2 / norme exacte",
                title=f"FE n = {nfe}, 256 pas",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        r.save_figure(
            fig,
            "night-reaction-moments",
            "La contraction coercive ne garantit pas la précision de la réaction. Les moments du champ complet restent proches du plancher FE sur les trois maillages et les trois intensités testées; une convergence générale n'est pas déduite de cette campagne.",
        )
        controls = [x["result"] for x in raw["reaction-moments-controls"]]
        if controls:
            discrepancies = []
            for x in controls:
                original_method = (
                    "fe-only" if x["method"] == "fe-projected" else "coercive-lift"
                )
                match = next(
                    y
                    for y in moments
                    if (y["nfe"], y["steps"], y["strength"], y["method"])
                    == (x["nfe"], x["steps"], x["strength"], original_method)
                )
                discrepancies.append(abs(x["relative_l2"] / match["relative_l2"] - 1))
            r.paragraph(
                f"Une ablation remplace l'exponentielle du générateur FE par la projection FE de l'exponentielle physique, sans traiter spatialement la mémoire. Sur les {len(controls)} essais, cette modification change les erreurs relatives d'au plus {max(discrepancies):.3g} en proportion : elle n'explique pas le gain de la nouvelle opération. Les temps sont ceux d'une exécution par configuration. La construction des masses pondérées ajoute environ 0,5 s pour n = 64 et 1,8 s pour n = 128; les moments complets coûtent davantage par pas que les deux modèles incomplets."
            )
        validation = args.input / "reaction-moments.validation.json"
        if validation.exists():
            checks = json.loads(validation.read_text())["results"]
            r.paragraph(
                f"Les {len(checks)} contrôles denses GPU aux degrés 1, 2 et 4 vérifient les Grams pondérés, la limite de réaction uniforme, l'observation FE et Pythagore. Le défaut de Pythagore maximal est {max(x['pythagoras_error'] for x in checks):.3g}; la variation maximale du Gram à la subdivision de quadrature est {max(x['quadrature_error'] or 0 for x in checks):.3g}. Compute Sanitizer ne signale aucune erreur sur cette validation."
            )
        refined = [x["result"] for x in raw["reaction-moments-quadrature"]]
        if refined:
            x = refined[0]
            match = next(
                y
                for y in moments
                if (y["nfe"], y["steps"], y["strength"], y["method"])
                == (x["nfe"], x["steps"], x["strength"], x["method"])
            )
            r.paragraph(
                f"Sur l'EDP à κ = 100, n = 64 et 256 pas, doubler les subdivisions des intégrales pondérées change l'erreur relative de {abs(x['relative_l2'] - match['relative_l2']):.3g} en valeur absolue."
            )
        name = "night-reaction-fields-64-256-100"
        if (args.output / (name + ".pdf")).exists():
            r.figure(
                name,
                "Champs calculés et erreurs, normalisés par le maximum exact dans la fenêtre affichée. Les deux cartes supérieures utilisent la même échelle; la coupe emploie une échelle logarithmique symétrique. Les trois cartes d'erreur ont une échelle logarithmique commune. Ce calcul pour la visualisation est séparé des mesures de temps.",
            )
    spectral = [x["result"] for x in raw["spectral-reaction"]]
    if spectral:
        r.section("Réaction couplée de quatre espèces")
        r.paragraph(
            "L'extension utilise la réaction κ(1 + 0,5 sin(2πx)) B, où B est le Laplacien d'un cycle à quatre espèces. Son spectre 0, 2, 2, 4 sépare un mode conservé et trois modes dissipatifs. Les blocs de moments pondérés sont réutilisés pour les valeurs propres répétées; une seule projection spatiale partagée traite les quatre composantes. La somme des fractions est un invariant de la réaction et du transport."
        )
        r.paragraph(
            "Pour mesurer correctement l'approche d'un équilibre non nul, le calcul évolue les écarts à (1/4, 1/4, 1/4, 1/4). Leur somme initiale est nulle. La référence et le plancher FE sont calculés directement sur ces écarts, sans soustraction de deux références proches de 1/4. L'erreur relative utilise la norme finale de l'écart à l'équilibre. Les comparateurs traitent soit la seule observation FE, soit cette observation et un amortissement uniforme spécifique à chaque mode dissipatif."
        )
        names = {
            "fe-only": "FE seule",
            "modal-lift": "Mémoire amortie par mode",
            "ambient-moments": "Moments complets partagés",
        }
        rows = [x for x in spectral if x["nfe"] == 64 and x["steps"] == 256]
        r.table(
            [
                "κ",
                "Modèle",
                "L2 relative",
                "Plancher relatif",
                "Défaut de somme L2",
                "Évolution (s)",
            ],
            [
                [
                    x["strength"],
                    names[x["method"]],
                    x["relative_l2"],
                    x["relative_fe_floor"],
                    x["fe_closure_l2"],
                    x["evolution_seconds"],
                ]
                for x in rows
            ],
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        for ax, nfe in zip(axes, (32, 64)):
            for method in names:
                selected = sorted(
                    [
                        x
                        for x in spectral
                        if x["nfe"] == nfe
                        and x["steps"] == 256
                        and x["method"] == method
                    ],
                    key=lambda x: x["strength"],
                )
                ax.semilogy(
                    [x["strength"] for x in selected],
                    [x["relative_l2"] for x in selected],
                    "o-",
                    label=names[method],
                )
            ax.semilogy(
                [x["strength"] for x in selected],
                [x["relative_fe_floor"] for x in selected],
                "k--",
                label="Plancher FE",
            )
            ax.set(
                xlabel="Intensité κ",
                ylabel="Erreur / norme de l'écart exact",
                title=f"FE n = {nfe}, 256 pas",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        r.save_figure(
            fig,
            "night-spectral-reaction",
            "La réaction du champ complet reste proche du plancher FE pour ce système conservant la somme des fractions. Elle coûte ici environ dix fois plus par évolution que la réaction FE seule; les trois calculs ne représentent pas avec la même précision la physique de la mémoire.",
        )
        check_file = args.input / "spectral-reaction.validation.json"
        if check_file.exists():
            checks = json.loads(check_file.read_text())["results"]
            shared = [x for x in checks if x["shared"]]
            independent = [x for x in checks if not x["shared"]]
            r.paragraph(
                f"Les {len(checks)} contrôles denses donnent un écart de covariance maximal de {max(x['covariance_error'] for x in shared):.3g} pour la projection partagée, contre {max(x['covariance_error'] for x in independent):.3g} pour les projections séparées. Le défaut de somme dans le champ auxiliaire atteint {max(x['auxiliary_closure_l2'] for x in independent):.3g} avec les projections séparées, contre {max(x['auxiliary_closure_l2'] for x in shared):.3g} avec l'espace partagé. Les deux variantes ont pourtant la même observation FE immédiate à la précision des arrondis. Compute Sanitizer ne signale aucune erreur sur ces contrôles."
            )
        r.paragraph(
            f"Dans les {len(spectral)} essais d'EDP, le défaut maximal de somme FE est {max(x['fe_closure_l2'] for x in spectral):.3g}. Les fractions échantillonnées restent positives, avec un minimum de {min(x['minimum_fraction_sampled'] for x in spectral):.3g}; aucune garantie générale de positivité n'en découle. Cette expérience porte sur une matrice de réaction symétrique constante multipliée par un taux spatial, et ne couvre pas un réseau chimique non linéaire général."
        )
    projection = [x["result"] for x in raw["reaction-projection"]]
    if projection:
        r.section("Compression des moments face à une projection ambiante complète")
        r.paragraph(
            "Un oracle supplémentaire résout par diagonalisation GPU le Gram conjoint complet de V et W, avec un seuil relatif de rang de 10⁻¹². Il est comparé aux petits espaces cachés sur des états aléatoires à quatre composantes. Ce contrôle mesure la perte de projection pour une réaction seule; il ne mesure pas une erreur d'EDP. La référence complète est elle-même soumise au seuil de rang."
        )
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
        for ax, degree in zip(axes, (1, 2, 4)):
            for shared in (False, True):
                rows = sorted(
                    [
                        x
                        for x in projection
                        if x["degree"] == degree
                        and x["duration"] == 1
                        and x["shared"] == shared
                    ],
                    key=lambda x: x["directions_per_field"],
                )
                ax.semilogy(
                    [x["directions_per_field"] for x in rows],
                    [sum(x["compressed_projection_loss"]) for x in rows],
                    "o-",
                    label="Partagé" if shared else "Séparé",
                )
            ax.axhline(
                sum(rows[0]["full_projection_loss"]),
                color="black",
                linestyle="--",
                label="Projection complète",
            )
            ax.set(
                xlabel="Directions par composante",
                ylabel="Distance au carré à gq",
                title=f"Degré {degree}, τ = 1",
            )
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
        r.save_figure(
            fig,
            "night-reaction-projection",
            "Enrichir la projection retient davantage de mémoire sur une même entrée. Le calcul partagé réunit les directions des quatre composantes. La distance résiduelle à la projection complète quantifie la limite d'une compression de petite dimension.",
        )
    residual = [x["result"] for x in raw["residual-return"]]
    if residual:
        r.section("Choisir les directions à partir d'un résidu")
        r.paragraph(
            "Deux variantes remplacent le moment h servant à générer les directions par h − Kd, ou par h − σKd avec σ ajusté au mieux dans la direction entrante. La projection finale conserve sa définition orthogonale. Le deuxième choix utilise un seul coefficient commun lorsque les champs partagent l'espace, afin de préserver la covariance des composantes. Les contrôles locaux de réaction montrent des réductions de distance au champ cible; elles ne se traduisent pas par un avantage systématique sur le retour avec supports mobiles."
        )
        r.table(
            ["T aller", "Pas", "Contrôle", "Direction", "L2 maximale", "Évolution (s)"],
            [
                [
                    x["half_duration"],
                    x["steps_per_leg"],
                    x["check_every"],
                    x["krylov_seed"],
                    max(x["l2"]),
                    x["evolution_seconds"],
                ]
                for x in residual
            ],
        )
    incremental = [x["result"] for x in raw["increment-physical"]]
    reanchored = [x["result"] for x in raw["increment-reanchor"]]
    if incremental:
        r.section(
            "Un défaut local précis peut masquer une représentation mal conditionnée"
        )
        r.paragraph(
            "Une autre variante calcule les moments de (g−1)q avec expm1, projette cet incrément et l'ajoute à l'état précédent. En arithmétique exacte, cette opération équivaut à projeter gq si l'ancien état appartient à l'espace retenu. La troncature numérique du Gram peut supprimer cette condition. Le calcul local du défaut est néanmoins très précis : sur 36 contrôles denses, jusqu'à un sous-pas de 10⁻¹², l'oracle indépendant retrouve l'énergie et le défaut de l'incrément avec une erreur relative inférieure à 2 × 10⁻¹⁴; le sous-pas nul conserve les coefficients bit à bit."
        )
        chosen = [
            x
            for x in moments + incremental + reanchored
            if x["nfe"] in (64, 128)
            and x["steps"] == 256
            and x["strength"] == 100
            and x["method"]
            in ("ambient-moments", "increment-moments", "increment-reanchor")
        ]
        r.table(
            ["FE n", "Opération", "Erreur relative", "Plancher FE"],
            [
                [x["nfe"], x["method"], x["relative_l2"], x["relative_fe_floor"]]
                for x in chosen
            ],
        )
        r.paragraph(
            "La mise à jour additive échoue pourtant lors de la réaction forte répétée. Les coefficients de l'ancre FE et du détail auxiliaire restent grands alors que leur somme physique devrait presque s'annuler. Le bilan d'énergie obtenu par différence perd alors lui-même des chiffres significatifs. Une variante utilisant les mêmes moments d'incrément, mais projetant à nouveau le champ complet, retrouve le niveau d'erreur de la méthode initiale. Cette variante remet les coefficients à l'échelle du champ qui subsiste et supprime les directions rejetées par le Gram."
        )
        conditions = [x["result"] for x in raw["increment-conditioning"]]
        if conditions:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
            for ax, record in zip(axes, conditions):
                history = record["history"]
                for key, label in (
                    ("anchor_energy", "Norme² de l'ancre"),
                    ("detail_energy", "Norme² du détail"),
                    ("visible_energy", "Norme² FE observée"),
                    ("ambient_energy", "Énergie par différence"),
                ):
                    ax.semilogy(
                        [x["time"] for x in history],
                        [x[key] if x[key] > 0 else float("nan") for x in history],
                        label=label,
                    )
                ax.set(
                    xlabel="Temps",
                    ylabel="Norme au carré",
                    title=(
                        "Champ complet reprojeté"
                        if record["reanchor"]
                        else "Incrément ajouté"
                    ),
                )
                ax.legend(fontsize=7)
                ax.grid(alpha=0.2)
            r.save_figure(
                fig,
                "night-increment-conditioning",
                "FE n = 128, auxiliaire n = 256, κ = 100. Dans la mise à jour additive, l'ancre et le détail conservent chacun une norme au carré proche de 0,1. La formule d'énergie soustrait ces termes et ne fournit plus une mesure fiable du très petit champ physique. Seules ses valeurs positives sont tracées. La reprojection complète maintient les coefficients à l'échelle du champ restant. Ces répétitions instrumentées sont exclues des temps comparatifs.",
            )
        r.paragraph(
            "Ce résultat négatif interdit d'assimiler automatiquement une formule stable pour le défaut local à une évolution robuste. L'identité de projection et la coercivité pure demandent que l'ancienne composante cachée soit retenue; la note explicite le terme supplémentaire lorsque cette hypothèse est perdue. La projection du champ complet reste la méthode de référence."
        )
    joint = [x["result"] for x in raw["joint-return"]]
    if joint:
        r.section("Association de la projection enrichie et de la phase minimax")
        r.paragraph(
            "La projection enrichie et la minimisation du déplacement périodique sont ensuite utilisées ensemble. Les essais courts reprennent les trois fréquences de contrôle. Deux essais plus longs sur FE n=256 et auxiliaire n=128 comparent la projection enrichie avec coupure fixe et avec phase minimax. Les ingrédients ont des contrôles séparés; cette campagne examine leur interaction."
        )
        r.table(
            [
                "T aller",
                "FE n",
                "Phase",
                "Pas / contrôle",
                "Changements",
                "L2 max",
                "Temps (s)",
            ],
            [
                [
                    x["half_duration"],
                    x["nfe"],
                    "minimax" if x["floating_phase"] else "fixe",
                    f"{x['steps_per_leg']} / {x['check_every']}",
                    x["changes"],
                    max(x["l2"]),
                    x["setup_seconds"] + x["evolution_seconds"],
                ]
                for x in joint
            ],
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for rows, label in (
            (
                [
                    x["result"]
                    for x in raw["warm-return"]
                    if x["result"]["method"] == "projected"
                ],
                "Filtre, coupe fixe",
            ),
            (minimax, "Filtre, phase minimax"),
            (
                [
                    x
                    for x in extended
                    if x["nfe"] == 128
                    and x["naux"] == 64
                    and x["half_duration"] == 0.25
                    and x["method"] == "chebyshev"
                ],
                "Chebyshev, coupe fixe",
            ),
            ([x["result"] for x in raw["galerkin"]], "Galerkin, coupe fixe"),
            (
                [x for x in joint if x["half_duration"] == 0.25],
                "Galerkin, phase minimax",
            ),
        ):
            if rows:
                ax.scatter(
                    [x["setup_seconds"] + x["evolution_seconds"] for x in rows],
                    [max(x["l2"]) for x in rows],
                    label=label,
                )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Temps de préparation et d'évolution (s)",
            ylabel="Erreur L2 maximale au retour",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        r.save_figure(
            fig,
            "night-joint",
            "Comparaison exploratoire des configurations courtes, à trois fréquences de contrôle. Chaque point correspond à une exécution; les trajectoires de supports sont propres à chaque méthode. Cette figure ne remplace pas la comparaison à géométrie identique.",
        )
        r.paragraph(
            "Le gain obtenu par Galerkin face au filtre tronqué à 512 pas ne suffit pas à établir un avantage de coût : Chebyshev à 128 pas donne déjà une erreur proche avec beaucoup moins de contrôles. Les paramètres de transport et d'adaptation doivent être optimisés ensemble avant une comparaison à précision imposée."
        )
    budgeted = [x["result"] for x in raw["budgeted-return"] + raw["budgeted-remaining"]]
    if budgeted:
        r.section("Contrôle du déplacement par un budget de transfert")
        r.paragraph(
            "Le contrôleur propose les mêmes partitions d'énergie, puis réduit leur déplacement par facteurs de deux jusqu'à respecter un budget de distance de mémoire. La projection augmentée relie cette distance à la perte d'énergie. Le budget total est réparti entre les occasions de changement; neuf essais au plus sont autorisés, après quoi l'ancien support est conservé. Une marge empirique pour les arrondis accompagne le diagnostic."
        )
        r.paragraph(
            "Dans le modèle ambiant exact, la somme des distances de compression borne leur effet cumulé pour la même suite d'opérateurs de transport figée. Ce budget ne borne pas l'erreur totale vis-à-vis de l'EDP. En particulier, ralentir le maillage peut augmenter l'erreur du transport; les courbes examinent ce compromis."
        )
        r.paragraph(
            "L'allocation uniforme échoue dans ces essais : elle peut empêcher tous les changements et laisser un budget inutilisé, avec une erreur physique finale élevée. Le contrôle du défaut de compression seul ne sélectionne donc pas un support adapté à l'advection."
        )
        if raw["budgeted-remaining"]:
            r.paragraph(
                "Une ablation autorise chaque changement à utiliser toute la fraction encore disponible du budget total. Elle isole le rôle de l'allocation uniforme; le même suivi de la somme des distances est conservé."
            )
        r.table(
            [
                "Allocation",
                "Pas / contrôle",
                "Budget",
                "Dépensé",
                "Changements",
                "Essais",
                "L2 max",
                "Temps (s)",
            ],
            [
                [
                    x.get("budget_allocation", "uniform"),
                    f"{x['steps_per_leg']} / {x['check_every']}",
                    x["remap_budget"],
                    x["remap_budget_used"],
                    x["changes"],
                    sum(len(h["transfer_attempts"]) for h in x["history"]),
                    max(x["l2"]),
                    x["evolution_seconds"],
                ]
                for x in budgeted
            ],
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for x in budgeted:
            label = f"{x.get('budget_allocation', 'uniform')}, B={x['remap_budget']}, {x['steps_per_leg']} pas"
            times = [
                (h["leg"] + h["step"] / x["steps_per_leg"]) * x["half_duration"]
                for h in x["history"]
            ]
            axes[0].plot(
                times, [h["remap_budget_used"] for h in x["history"]], label=label
            )
            selected = [h for h in x["history"] if h["transfer_fraction"] is not None]
            axes[1].plot(
                [
                    (h["leg"] + h["step"] / x["steps_per_leg"]) * x["half_duration"]
                    for h in selected
                ],
                [h["transfer_fraction"] for h in selected],
                ".",
                label=label,
            )
        axes[0].set(
            xlabel="Temps du parcours aller-retour",
            ylabel="Somme des distances de transfert",
        )
        axes[1].set(
            xlabel="Temps du parcours aller-retour",
            ylabel="Fraction acceptée du déplacement proposé",
            ylim=(-0.03, 1.03),
        )
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        r.save_figure(
            fig,
            "night-remap-budget",
            "La colonne de droite expose les réductions et rejets du déplacement. Le respect du budget porte sur la mesure de compression; l'erreur L2 physique est présentée séparément dans le tableau.",
        )
    sources = [x["result"] for x in raw["source-full"] or raw["source-smoke"]]
    if sources:
        r.section("Apparition d'une structure par une source FE")
        r.paragraph(
            "Une nouvelle bosse centrée en (0,8; 0,75) est ajoutée par une source FE impulsionnelle à t=0,125, après un début de transport d'une autre bosse. Le maillage triangulaire est perturbé, sans graduation. L'adaptation compare l'énergie de l'état auxiliaire à celle du champ FE observé. Le second moniteur peut voir une structure créée par la physique FE avant qu'elle soit bien représentée dans l'auxiliaire. La référence transporte séparément les deux profils analytiques."
        )
        r.paragraph(
            "La source injecte de l'énergie. Son travail est mesuré par la différence d'énergie FE, et l'identité entre ce travail et la variation de l'énergie augmentée est contrôlée. Le bilan de masse ajoute la masse de la source. Une contraction globale n'est pas revendiquée à travers cette impulsion."
        )
        r.table(
            ["FE n", "Aux. n", "Largeur", "Moniteur", "L2", "Plancher FE"],
            [
                [
                    x["nfe"],
                    x["naux"],
                    x["width"],
                    x["monitor"],
                    max(x["l2"]),
                    max(x["best_fe_l2"]),
                ]
                for x in sources
            ],
        )
        if (args.output / "night-source-fields.pdf").exists():
            r.figure(
                "night-source-fields",
                "Structure ajoutée par la source FE, observée à la fin du transport. Même maillage FE n=256 et même moniteur FE; seule la résolution auxiliaire passe de 32 à 64. Les erreurs ponctuelles partagent une échelle logarithmique. Le panneau de projection mesure la limite FE locale.",
            )
    source_refinement = [x["result"] for x in raw["source-refinement"]]
    if source_refinement:
        r.paragraph(
            "Un raffinement emboîté peut être déclenché par la part de la source FE absente de l'auxiliaire. Après le changement de partition à l'injection, le diagnostic est la racine de la différence entre l'énergie FE de la source et l'énergie de sa projection auxiliaire. Les intervalles sont bisectés jusqu'à satisfaction du seuil ou épuisement du budget n=128. La mémoire est transférée par l'injection emboîtée, dont la perte d'énergie est contrôlée. Ce seuil porte sur la source projetée à cet instant; il ne borne pas l'erreur EDP finale."
        )
        r.table(
            [
                "Largeur",
                "Aux. initial",
                "Aux. final",
                "Seuil",
                "Atteint",
                "L2 max",
                "Temps (s)",
            ],
            [
                [
                    x["width"],
                    x["naux"],
                    x["final_naux"],
                    x["source_tolerance"],
                    (
                        (
                            "oui"
                            if x["source_resolution"][-1]["tolerance_met"]
                            else "non"
                        )
                        if x["source_resolution"]
                        else "fixe"
                    ),
                    max(x["l2"]),
                    x["setup_seconds"] + x["evolution_seconds"],
                ]
                for x in source_refinement
            ],
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, width in zip(axes, (0.003, 0.006)):
            controls = [
                x
                for x in sources
                if x["nfe"] == 256 and x["monitor"] == "fe" and x["width"] == width
            ]
            controls += [
                x
                for x in source_refinement
                if x["width"] == width and x["source_tolerance"] is None
            ]
            ax.loglog(
                [x["setup_seconds"] + x["evolution_seconds"] for x in controls],
                [max(x["l2"]) for x in controls],
                "o",
                label="Résolution auxiliaire fixe",
            )
            for x in controls:
                ax.annotate(
                    str(x["naux"]),
                    (x["setup_seconds"] + x["evolution_seconds"], max(x["l2"])),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
            adaptive = [
                x
                for x in source_refinement
                if x["width"] == width and x["source_tolerance"] is not None
            ]
            ax.loglog(
                [x["setup_seconds"] + x["evolution_seconds"] for x in adaptive],
                [max(x["l2"]) for x in adaptive],
                "s",
                label="Raffinement à l'injection",
            )
            if controls:
                ax.axhline(
                    max(controls[0]["best_fe_l2"]),
                    color="black",
                    linestyle="--",
                    label="Plancher FE",
                )
            ax.set(
                title=f"Largeur de source {width}",
                xlabel="Temps complet (s)",
                ylabel="Erreur L2",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        r.save_figure(
            fig,
            "night-source-refinement",
            "Coût de préparation et d'évolution. Les simulations à résolution fixe gardent un nombre de cellules constant mais leurs partitions se déplacent. Chaque point représente une exécution; aucun gain robuste de performance n'est conclu de cette seule mesure.",
        )
    diffusion = [x["result"] for x in raw["diffusion"]]
    if diffusion:
        r.section("Diffusion implicite dans l'espace FE")
        r.paragraph(
            "Une diffusion SIPG est assemblée sur les triangles perturbés DG P4. Chaque demi-pas Crank–Nicolson est résolu par gradient conjugué préconditionné sur GPU; les blocs de raideur et le préconditionneur diagonal sont réutilisés. Le transport intermédiaire est une translation périodique de vitesse (1; 0,37). Une solution analytique de l'équation advection–chaleur fournit la référence à t=0,23, avec diffusivité 0,002. Le contrôle efface la mémoire après chaque transport."
        )
        r.paragraph(
            "L'opérateur symétrique, son noyau constant, son spectre et la résolution CN sont contrôlés sur de petits maillages par diagonalisation GPU indépendante du solveur itératif. La pénalité vaut huit fois (p+1) au carré, multiplié par l'inverse de la plus petite hauteur adjacente. Ces contrôles ne constituent pas une preuve de coercivité pour des maillages arbitraires. L'ordre deux de CN ne suffit pas à établir celui du couplage avec projections."
        )
        control_file = args.input / "diffusion.validation.json"
        if control_file.exists():
            heat_controls = json.loads(control_file.read_text()).get("heat_only", [])
            r.paragraph(
                "Un contrôle sans advection vérifie aussi la décroissance analytique du mode cos(2πx) sin(2πy), jusqu'à t=0,2 avec 32 pas CN. Ce test contrôle le signe et la normalisation de la diffusion indépendamment du transfert semi-lagrangien."
            )
            r.table(
                ["FE n", "Erreur L2 chaleur seule", "Plancher FE"],
                [[x["nfe"], max(x["l2"]), max(x["best_fe_l2"])] for x in heat_controls],
            )
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        for ax, nfe in zip(axes, (8, 16, 32)):
            selected = [x for x in diffusion if x["nfe"] == nfe and x["fields"] == 1]
            for method in ("memory", "reset"):
                group = sorted(
                    [x for x in selected if x["method"] == method],
                    key=lambda x: x["steps"],
                )
                if group:
                    ax.loglog(
                        [x["steps"] for x in group],
                        [max(x["l2"]) for x in group],
                        "o-",
                        label=method,
                    )
            if selected:
                ax.axhline(
                    max(selected[0]["best_fe_l2"]),
                    color="black",
                    linestyle="--",
                    label="Plancher FE",
                )
            ax.set(
                title=f"FE {nfe}, auxiliaire {2*nfe}",
                xlabel="Nombre de pas",
                ylabel="Erreur L2",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        r.save_figure(
            fig,
            "night-diffusion",
            "Diffusion FE implicite entre transports auxiliaires. La solution analytique intègre simultanément translation et élargissement du profil; le plancher FE est calculé par projection indépendante.",
        )
        r.table(
            ["FE n", "Champs", "Pas", "Méthode", "L2 max", "CG max", "Temps (s)"],
            [
                [
                    x["nfe"],
                    x["fields"],
                    x["steps"],
                    x["method"],
                    max(x["l2"]),
                    x["cg_iterations_max"],
                    x["setup_seconds"] + x["evolution_seconds"],
                ]
                for x in diffusion
            ],
        )
        r.paragraph(
            f"Résidu relatif maximal des résolutions : {max(x['cg_residual_max'] for x in diffusion):.3g}. Écart absolu maximal de masse : {max(h['mass_error'] for x in diffusion for h in x['history']):.3g}. Le maximum du rapport d'énergie augmentée observé après un pas vaut {max(h['energy_ratio'] for x in diffusion for h in x['history']):.6g}. Les temps comprennent un assemblage par cas; aucun graphe CUDA ni solveur multi-second-membre n'est employé pour CG."
        )
        if (args.output / "night-diffusion-fields.pdf").exists():
            r.figure(
                "night-diffusion-fields",
                "Champ FE après translation et diffusion, avec et sans mémoire. Maillage perturbé n=16, auxiliaire n=32 et 64 pas. Les erreurs locales sont affichées à échelle commune; leurs valeurs L2 sont intégrées indépendamment sur tout le domaine.",
            )
    products = [x["result"] for x in raw["products"] or raw["product-pilot"]]
    if products:
        r.section("Exploration : vitesses issues de produits de fonctions de courant")
        r.paragraph(
            "Une fonction de courant somme de produits a(y)b(x) transforme l'advection en somme de commutateurs de cisaillements. Le prototype réutilise une formule publiée de Casas, Escorihuela-Tomàs et Moreno Casares (2025), PCP16. Ses seize cisaillements Fourier sont unitaires; son ordre cinq dans la racine du pas correspond à l'ordre deux dans le temps physique, à grille fixée. Les coefficients et cette théorie de composition sont attribués aux auteurs. L'intérêt recherché ici est une extension des vitesses admissibles dans le couplage FE, dont la nouveauté et l'avantage de coût ne sont pas établis."
        )
        r.paragraph(
            "Les tests emploient un ou deux tourbillons gaussiens périodiques, dont la fonction de courant possède de nombreux modes de Fourier mais peu de produits séparables. Un oracle matriciel GPU contrôle le commutateur, l'unitarité, les constantes, l'inverse et le défaut temporel. Une intégration indépendante de caractéristiques contrôle l'erreur physique. Une erreur nodale RMS est distinguée de l'erreur FE intégrée."
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for rank, ax in zip((1, 2), axes):
            for n in (127, 255, 511):
                for method in ("group4", "pcp16"):
                    selected = sorted(
                        [
                            x
                            for x in products
                            if x["rank"] == rank
                            and x["n"] == n
                            and x["method"] == method
                            and x["nfe"] is None
                        ],
                        key=lambda x: x["steps"],
                    )
                    if selected:
                        ax.loglog(
                            [x["steps"] for x in selected],
                            [max(x["l2"]) for x in selected],
                            "o-",
                            label=f"{method}, n={n}",
                        )
            ax.set(
                title=f"{rank} produit(s)",
                xlabel="Nombre de pas",
                ylabel="Erreur RMS nodale",
            )
            if ax.lines:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        r.save_figure(
            fig,
            "night-products",
            "La grille de 127 points peut imposer un plancher spatial. Les résultats nodaux exploratoires ne constituent pas un benchmark de performance FE ni une validation d'une vitesse FE arbitraire.",
        )
        fe_products = [x for x in products if x["nfe"] is not None]
        if fe_products:
            r.table(
                ["Produits", "FE n", "Pas", "L2 FE", "Plancher FE"],
                [
                    [
                        x["rank"],
                        x["nfe"],
                        x["steps"],
                        max(x["l2"]),
                        max(x["best_fe_l2"]),
                    ]
                    for x in fe_products
                ],
            )
    svd = [x["result"] for x in raw["svd"]]
    if svd:
        r.section("Fonction de courant échantillonnée et comparaison Fourier/RK4")
        r.paragraph(
            "Le vortex comporte un facteur exponentiel en cos(2π(x+y)), donc une séparation de rang fini n'est plus prescrite. Une SVD GPU de la fonction de courant sélectionne un préfixe selon l'erreur L2 de vitesse spectrale. L'orthogonalité permet de sommer exactement les énergies des termes rejetés. Ce choix ne minimise pas le rang en norme H1. Le comparateur Fourier utilise la forme antisymétrique de l'advection et RK4, avec un pas borné par la norme de son générateur. Les deux chemins utilisent des graphes CUDA et les mêmes grilles."
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for n, ax in zip((127, 255), axes):
            for label, method, tol in (
                ("RK4", "rk4", None),
                ("PCP16, tol. 0,01", "pcp16", 1e-2),
                ("PCP16, tol. 0,0001", "pcp16", 1e-4),
                ("PCP16, tol. 0,000001", "pcp16", 1e-6),
            ):
                selected = sorted(
                    [
                        x
                        for x in svd
                        if x["n"] == n
                        and x["method"] == method
                        and (tol is None or x["tolerance"] == tol)
                    ],
                    key=lambda x: x["evolution_seconds"],
                )
                if selected:
                    ax.loglog(
                        [x["setup_seconds"] + x["evolution_seconds"] for x in selected],
                        [max(x["l2"]) for x in selected],
                        "o-",
                        label=label,
                    )
            ax.set(
                title=f"Grille {n} × {n}",
                xlabel="Préparation + évolution (s)",
                ylabel="Erreur RMS nodale",
            )
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
        r.save_figure(
            fig,
            "night-svd-cost",
            "Comparaison exploratoire à une répétition, référence et grille initiale exclues des temps. Une erreur de vitesse SVD ne constitue pas à elle seule une borne sur l'erreur du champ transporté.",
        )
        r.table(
            [
                "Grille",
                "Méthode",
                "Rang",
                "Pas",
                "Erreur vitesse",
                "L2 champ",
                "Temps (s)",
            ],
            [
                [
                    x["n"],
                    x["method"],
                    (x["approximation"] or {}).get("rank", "—"),
                    x["steps"],
                    (x["approximation"] or {}).get("velocity_error_l2", "—"),
                    max(x["l2"]),
                    x["setup_seconds"] + x["evolution_seconds"],
                ]
                for x in svd
            ],
        )
        comparison_rk = [
            x
            for x in svd
            if x["n"] == 255 and x["method"] == "rk4" and x["rk_scale"] == 1
        ]
        comparison_pc = [
            x
            for x in svd
            if x["n"] == 255
            and x["method"] == "pcp16"
            and x["tolerance"] == 1e-6
            and x["steps"] == 256
        ]
        if comparison_rk and comparison_pc:
            a, b = comparison_rk[0], comparison_pc[0]
            ta, tb = (
                a["setup_seconds"] + a["evolution_seconds"],
                b["setup_seconds"] + b["evolution_seconds"],
            )
            r.paragraph(
                f"Sur ce vortex résolu, Fourier/RK4 donne une erreur {max(a['l2']):.4g} en {ta:.4g} s, contre {max(b['l2']):.4g} en {tb:.4g} s pour PCP16. Le second coûte {tb/ta:.3g} fois plus et reste moins précis dans cette comparaison. Les {b['shears_per_step']} cisaillements par pas rendent cette compression SVD non compétitive ici. Sa conservation d'énergie ne suffit pas à justifier sa poursuite comme voie de performance prioritaire."
            )
    compressible = [x["result"] for x in raw["compressible"]]
    if compressible:
        r.section("Dépôt forward conservatif sous compression")
        r.paragraph(
            "Le flot exact de la vitesse (0,2 sin(2πx), 0) transporte une densité jusqu'à t=1. La référence comprend le Jacobien inverse : l'équation conservative diffère ici de l'équation advective. Les intervalles de départ sont découpés aux préimages des arêtes d'arrivée avant quadrature de Gauss. Les matrices unidimensionnelles sont construites et appliquées sur GPU. Un raffinement de quadrature 16 vers 32 contrôle cette intégration non polynomiale; un calcul de valeurs singulières vérifie la borne de croissance dans la norme de masse sur le petit cas de contrôle."
        )
        r.paragraph(
            "Le couplage conserve le défaut FE initial tandis que le support DG évolue. Le contrôle remet la mémoire à zéro après chaque pas. La masse doit rester constante; l'énergie augmentée peut croître physiquement, avec une borne exp(2π × 0,2 × t) sur son rapport. Les maillages FE sont perturbés. Cette construction traite un flot séparable exact; une vitesse compressible FE générale n'est pas couverte."
        )
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, (nfe, naux) in zip(
            axes.ravel(), ((32, 32), (32, 64), (64, 32), (64, 64))
        ):
            entries = [x for x in compressible if x["nfe"] == nfe and x["naux"] == naux]
            for method in ("memory", "reset"):
                selected = sorted(
                    [x for x in entries if x["method"] == method],
                    key=lambda x: x["steps"],
                )
                if selected:
                    ax.loglog(
                        [x["steps"] for x in selected],
                        [max(x["l2"]) for x in selected],
                        "o-",
                        label=method,
                    )
            if entries:
                ax.axhline(
                    max(entries[0]["best_fe_l2"]),
                    color="black",
                    linestyle="--",
                    label="Projection FE",
                )
                ax.legend(fontsize=8)
            ax.set(
                title=f"FE n={nfe} · auxiliaire n={naux}",
                xlabel="Nombre de dépôts",
                ylabel="Erreur L2 FE",
            )
            ax.grid(alpha=0.2)
        r.save_figure(
            fig,
            "night-compressible",
            "Le flot exact supprime l'erreur d'intégration temporelle des trajectoires. Le nombre de dépôts change l'accumulation des erreurs de projection, et non une erreur RK sur les caractéristiques.",
        )
        r.paragraph(
            f"Sur les {len(compressible)} cas enregistrés, l'écart absolu maximal de masse vaut {max(x['mass_error'] for x in compressible):.3g}; le rapport d'énergie maximal vaut {max(x['energy_ratio'] for x in compressible):.6g}, pour une borne physique {compressible[0]['physical_energy_bound']:.6g}."
        )
    r.section("Raffinement : identité validée, contrôleur insuffisant")
    nested = json.loads((args.input / "nested-validation.json").read_text())["results"]
    r.table(
        ["Degré", "Raffinement", "Défaut Gram", "Défaut observation"],
        [
            [
                x["p"],
                "local" if x["local"] else "global",
                x["gram_error"],
                x["observation_error"],
            ]
            for x in nested
        ],
    )
    r.paragraph(
        "Le premier contrôleur raffine selon la partie du générateur FE non représentée par le support. Les sélections par énergie et par détail hiérarchique atteignent toutes deux la limite de 96 intervalles dès le deuxième contrôle. L'erreur finale reste proche de 0,148. La conservation de l'énergie lors du raffinement ne répare donc pas ce contrôleur : le budget est épuisé tôt, et aucun déraffinement ne libère les cellules derrière la structure. L'indicateur utilisé ne mesure qu'un terme du défaut de consistance."
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    refinements = []
    for name, label in (("refinement-01", "Énergie"), ("refinement-detail", "Détail")):
        file = args.input / (name + ".json")
        if file.exists():
            record = json.loads(file.read_text())["result"]
            refinements.append(record)
            t = [x["step"] * record["end"] / record["steps"] for x in record["history"]]
            axes[0].step(
                t, [x["n"] for x in record["history"]], where="post", label=label
            )
            axes[1].semilogy(
                t, [x["indicator"] for x in record["history"]], label=label
            )
    axes[0].set(
        xlabel="Temps physique",
        ylabel="Intervalles par direction",
        title="Budget épuisé précocement",
    )
    axes[1].set(
        xlabel="Temps physique",
        ylabel="Indicateur",
        title="Critère non satisfait après saturation",
    )
    axes[1].axhline(0.1, color="black", linestyle="--")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    r.save_figure(
        fig,
        "night-refinement",
        "Résultat négatif conservé : changer le classement des cellules ne suffit pas à rendre le contrôleur efficace.",
    )
    reference_file = args.input / "reference-relative.validation.json"
    if reference_file.exists():
        relative = json.loads(reference_file.read_text())["results"]
        r.section(
            "Stabilité par rapport à une référence FE, même avec un transfert non linéaire"
        )
        r.paragraph(
            "Une identité permet de dépasser la seule décroissance d'énergie. Si le transfert conserve l'observation FE et diminue l'énergie augmentée, sa distance à toute référence FE minimale diminue exactement de l'énergie dissipée, en norme au carré. Cette propriété reste vraie lorsque l'espace de compression dépend de l'état. Elle n'affirme pas la contraction entre deux états possédant des mémoires différentes et ne contredit donc pas le contre-exemple précédent."
        )
        r.paragraph(
            "La preuve résulte du développement des deux normes et de la conservation du terme croisé avec la référence. Si l'évolution entre transferts est non expansive entre deux états, la récurrence borne l'erreur finale par l'erreur initiale et la somme des défauts locaux de transport. Aucun terme supplémentaire proportionnel au nombre de changements de support n'apparaît. Les supports réalisés modifient cependant ces défauts : cette borne ne rend pas le choix du maillage indifférent."
        )
        r.table(
            ["Contrôles GPU", "Défaut relatif d'identité", "Excès relatif de distance"],
            [
                [
                    len(relative),
                    max(x["relative_identity_defect"] for x in relative),
                    max(x["relative_distance_excess"] for x in relative),
                ]
            ],
        )
        r.paragraph(
            "Les contrôles utilisent une réalisation dense indépendante de l'espace augmenté, les degrés FE 1, 2 et 4, trois mouvements, cinq transferts et trois références d'amplitudes différentes. Une borne de convergence en fonction du pas et des maillages requiert encore une estimation des défauts locaux des noyaux de transport."
        )
    graph = [
        x["result"] for name in ("graph-return", "graph-return-long") for x in raw[name]
    ]
    if graph:
        r.section("Mémoire à énergie positive par réflexion sur le graphe du couplage")
        r.paragraph(
            "Une seconde réalisation remplace les différences de normes mal conditionnées par une somme de deux normes positives. Elle construit un réflecteur orthogonal à partir du graphe de R, la projection FE vers le support auxiliaire. Les seules inversions portent sur I+R*R, dont le spectre appartient à [1,2]. Le transfert obtenu est linéaire et contractant pour des supports prescrits; il préserve l'observation FE et devient l'identité à support inchangé."
        )
        r.paragraph(
            "Cette construction change l'injection FE : son premier bloc devient 2R multiplié par l'inverse de I+R*R. Elle définit donc un autre couplage, auquel les résultats de réaction par moments physiques ne sont pas automatiquement applicables. L'inverse est évalué par un polynôme fixe de degré 16 ou 20 sur GPU; aucun critère d'arrêt dépendant des données n'intervient."
        )
        file = args.input / "graph.validation.json"
        if file.exists():
            controls = json.loads(file.read_text())["controls"]
            r.table(
                [
                    "Degré inverse",
                    "Erreur état / oracle",
                    "Défaut observation",
                    "Résidu inverse",
                ],
                [
                    [degree]
                    + [
                        max(x[key] for x in controls if x["inverse_degree"] == degree)
                        for key in (
                            "state_oracle_relative_error",
                            "observed_error",
                            "inverse_relative_residual",
                        )
                    ]
                    for degree in (12, 16, 20)
                ],
            )
        r.table(
            [
                "FE",
                "Durée aller",
                "Pas",
                "Contrôle",
                "Degré inverse",
                "L2 max.",
                "Coût (s)",
            ],
            [
                [
                    x["nfe"],
                    x["half_duration"],
                    x["steps_per_leg"],
                    x["check_every"],
                    x["graph_degree"],
                    max(x["l2"]),
                    x["setup_seconds"] + x["evolution_seconds"],
                ]
                for x in graph
            ],
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        baselines = [
            x["result"]
            for x in raw["residual-return"]
            if x["result"]["krylov_seed"] == "moment"
        ]
        for ax, duration in zip(axes, (0.25, 1.0)):
            for degree in (16, 20):
                xs = [
                    x
                    for x in graph
                    if x["half_duration"] == duration and x["graph_degree"] == degree
                ]
                ax.scatter(
                    [x["setup_seconds"] + x["evolution_seconds"] for x in xs],
                    [max(x["l2"]) for x in xs],
                    label=f"Graphe, degré {degree}",
                )
            xs = [x for x in baselines if x["half_duration"] == duration]
            ax.scatter(
                [x["setup_seconds"] + x["evolution_seconds"] for x in xs],
                [max(x["l2"]) for x in xs],
                marker="x",
                label="Moments Galerkin",
            )
            ax.set(
                xlabel="Coût complet (s)",
                ylabel="Erreur L2 maximale",
                title=f"Retour après deux trajets de durée {duration}",
            )
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        r.save_figure(
            fig,
            "night-graph-return",
            "Comparaison exploratoire à une exécution par configuration. Chaque méthode adapte son propre support; les tableaux précisent le pas et la fréquence de contrôle. L'amélioration du conditionnement ne garantit pas une meilleure précision de transport.",
        )
        frozen = [x["result"] for x in raw["graph-frozen"]]
        if frozen:
            r.table(
                [
                    "Supports prescrits",
                    "L2 max.",
                    "Écart FE / graphe 20",
                    "Transport (s)",
                    "Transferts (s)",
                ],
                [
                    [
                        x["method"],
                        max(x["l2"]),
                        max(x["difference_from_graph20"]),
                        x["advection_seconds"],
                        x["transfer_seconds"],
                    ]
                    for x in frozen
                ],
            )
            r.paragraph(
                "Cette dernière comparaison partage exactement 63 changements de support prescrits par les trajectoires matérielles. Le coût géométrique commun est exclu des deux dernières colonnes. Elle sépare les différences entre inverses polynomiaux des rétroactions du contrôleur sur la géométrie."
            )
    consistency = [
        x["result"]
        for name in (
            "remap-consistency",
            "remap-consistency-orders",
            "remap-consistency-fine",
            "remap-consistency-smallstep",
            "remap-consistency-interval",
        )
        for x in raw[name]
    ]
    if consistency:
        r.section("Translation exacte avec changements de support à chaque pas")
        r.paragraph(
            "Quatre champs sinus et cosinus sont transportés à la vitesse constante (1; 0,37) jusqu'au temps 0,2. Le maillage triangulaire est légèrement perturbé. Deux partitions auxiliaires, distantes d'au plus 0,24 cellule par direction, alternent à chaque pas. Le nombre de transferts augmente donc lorsque le pas diminue, sans changement de résolution spatiale. La référence FE exacte est obtenue par rotation des composantes initiales; la référence physique est analytique."
        )
        r.paragraph(
            "La norme du défaut local est mesurée indépendamment sur l'injection de cette référence, pour chacun des deux supports. La somme de ces défauts donne la borne issue de la preuve; aucun coefficient n'est ajusté à l'erreur finale. La norme de Frobenius sur les quatre champs sert à ce contrôle, tandis que les courbes d'erreur physique affichent le maximum des quatre normes L2. Les valeurs restent des calculs en double précision, sans certification par intervalles."
        )
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        colors = {"reset": "#777777", "galerkin": "#d56826", "graph": "#176a8a"}
        for method in colors:
            selected = [
                x
                for x in consistency
                if x["method"] == method
                and x["degree"] == 4
                and x["steps"] == 256
                and x["remap_every"] == 1
            ]
            selected.sort(key=lambda x: x["n"])
            axes[0].loglog(
                [x["n"] for x in selected],
                [max(x["l2"]) for x in selected],
                "o-",
                color=colors[method],
                label=method,
            )
            selected = [
                x
                for x in consistency
                if x["method"] == method
                and x["degree"] == 4
                and x["n"] == 32
                and x["remap_every"] == 1
            ]
            selected.sort(key=lambda x: x["steps"])
            axes[1].loglog(
                [x["steps"] for x in selected],
                [max(x["l2"]) for x in selected],
                "o-",
                color=colors[method],
                label=method,
            )
            axes[2].loglog(
                [x["steps"] for x in selected],
                [x["summed_defect_bound"] for x in selected],
                "o--",
                color=colors[method],
                label=method + " : borne",
            )
            axes[2].loglog(
                [x["steps"] for x in selected],
                [x["fe_reference_frobenius_error"] for x in selected],
                "o-",
                color=colors[method],
                label=method + " : erreur FE",
            )
        axes[0].set(
            xlabel="Résolution n",
            ylabel="Erreur L2 maximale",
            title="P4, 255 transferts",
        )
        axes[1].set(
            xlabel="Nombre de pas",
            ylabel="Erreur L2 maximale",
            title="P4, résolution 32",
        )
        axes[2].set(
            xlabel="Nombre de pas",
            ylabel="Norme sur les quatre champs",
            title="Borne par les défauts locaux",
        )
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        r.save_figure(
            fig,
            "night-remap-consistency",
            "L'erreur physique et la borne de consistance sont distinguées. Une augmentation du nombre de pas accroît ici le nombre de changements de support; elle ne constitue pas une étude temporelle à opérateurs de transfert fixés.",
        )
        r.table(
            [
                "Cas contrôlés",
                "Rapport erreur / borne max.",
                "Défaut masse max.",
                "Excès énergie max.",
            ],
            [
                [
                    len(consistency),
                    max(x["bound_ratio"] for x in consistency),
                    max(x["mass_error"] for x in consistency),
                    max(x["energy_ratio"] - 1 for x in consistency),
                ]
            ],
        )
        r.table(
            ["Degré FE", "Méthode", "n", "Pas", "L2 max.", "Plancher FE"],
            [
                [
                    x["degree"],
                    x["method"],
                    x["n"],
                    x["steps"],
                    max(x["l2"]),
                    max(x["best_fe_l2"]),
                ]
                for x in consistency
                if x["n"] in (32, 64) and x["steps"] == 256 and x["remap_every"] == 1
            ],
        )
    positive_file = args.input / "residual-energy.validation.json"
    if positive_file.exists():
        positive_controls = json.loads(positive_file.read_text())["results"]
        r.section("Mesurer les petites énergies sans soustraction de grands termes")
        r.paragraph(
            "Un nouveau diagnostic CUDA intègre directement le carré de la différence FE–auxiliaire sur leurs intersections, avec des poids positifs. L'ordre de quadrature intègre exactement le carré du polynôme Pp–Qq sur chaque intersection affine. Le résidu est formé aux points de quadrature avant d'être élevé au carré. Cette mesure évite la soustraction de deux énergies presque égales; elle ne modifie pas l'évolution numérique."
        )
        r.paragraph(
            f"Les {len(positive_controls)} contrôles couvrent P1, P2 et P4, des partitions uniformes et décalées. Sur des champs aléatoires, l'écart relatif maximal à l'identité bilinéaire indépendante vaut {max(x['random_relative_error'] for x in positive_controls):.3g}. Pour deux constantes séparées de 1e-12, l'énergie physique est de l'ordre de 1e-24 : la quadrature positive la résout, alors que la formule soustractive retourne une valeur de l'ordre de l'arrondi absolu, parfois négative."
        )
        conditioning = [x["result"] for x in raw["increment-conditioning-positive"]]
        if conditioning:
            r.table(
                [
                    "Mise à jour",
                    "Énergie positive",
                    "Énergie soustractive",
                    "Énergie FE visible",
                ],
                [
                    [
                        "Reprojection" if x["reanchor"] else "Addition",
                        x["history"][-1]["positive_ambient_energy"],
                        x["history"][-1]["ambient_energy"],
                        x["history"][-1]["visible_energy"],
                    ]
                    for x in conditioning
                ],
            )
            r.paragraph(
                "La mesure positive confirme une mémoire résiduelle physique dans le calcul additif à réaction forte : ce n'est donc pas seulement un défaut du diagnostic soustractif. La reprojection du champ complet ramène aussi cette énergie cachée à l'échelle du champ amorti. Le seuil de rang peut exclure une direction de faible énergie absolue qui reste déterminante relativement à une solution presque nulle."
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            for x in conditioning:
                history = [h for h in x["history"] if "positive_ambient_energy" in h]
                label = (
                    "Reprojection complète"
                    if x["reanchor"]
                    else "Addition d'incréments"
                )
                ax.semilogy(
                    [h["time"] for h in history],
                    [h["positive_ambient_energy"] for h in history],
                    "o-",
                    markersize=3,
                    label=label,
                )
            ax.set(
                xlabel="Temps",
                ylabel="Énergie physique par quadrature positive",
                title="Réaction forte : mémoire réellement persistante",
            )
            ax.legend()
            ax.grid(alpha=0.2)
            r.save_figure(
                fig,
                "night-positive-energy",
                "Même problème de réaction de force 100 et même discrétisation. Les diagnostics sont ajoutés à des répétitions instrumentées, exclues des comparaisons de temps.",
            )
    r.section("Contrôles et limites pour un article")
    r.paragraph(
        "La borne d'erreur pour des supports variables repose sur la conservation de l'observation et la décroissance de l'énergie lors du transfert, ainsi que sur la non-expansivité de l'évolution entre transferts. Les défauts locaux de transport restent dans la borne, notamment le commutateur entre projection et générateur. Ce résultat conditionnel ne fournit ni un ordre de convergence pour les noyaux actuels ni une garantie d'efficacité du contrôleur."
    )
    expected = {
        "matched": 18,
        "adaptive-physics": 24,
        "adaptive-chebyshev": 12,
        "moving-separated": 18,
        "adaptive-separated": 12,
        "memory-stress": 3,
        "separated-refinement": 12,
        "roundtrip-smoke": 6,
        "roundtrip-long": 48,
        "auxiliary-return": 8,
        "source-full": 16,
        "source-refinement": 6,
        "products": 32,
        "monitor-stationary": 12,
        "monitor-return": 12,
        "svd": 16,
        "compressible": 32,
        "warm-return": 6,
        "material-monitor": 4,
        "monitor-fe": 3,
        "material-phase": 2,
        "phase-return": 3,
        "galerkin": 3,
        "budgeted-return": 4,
        "budgeted-remaining": 2,
        "joint-return": 5,
        "frozen-transfer": 8,
        "krylov": 16,
        "krylov-tuned": 4,
        "krylov-fine": 8,
        "block-frozen": 5,
        "gram": 12,
        "fast-return": 8,
        "system-coupling": 12,
        "mixture": 12,
        "relaxation": 36,
        "reaction-moments": 36,
        "reaction-moments-controls": 24,
        "reaction-moments-fine": 27,
        "reaction-moments-quadrature": 1,
        "reaction-projection": 90,
        "spectral-reaction": 36,
        "tiled-return-pilot": 1,
        "long-return-benchmark": 15,
        "residual-return": 12,
        "reaction-increment-projection": 90,
        "reaction-residual-projection": 90,
        "increment-physical": 18,
        "increment-reanchor": 2,
        "increment-conditioning": 2,
        "increment-conditioning-positive": 2,
        "graph-return": 6,
        "graph-return-long": 2,
        "graph-frozen": 4,
        "remap-consistency": 27,
        "remap-consistency-orders": 24,
        "remap-consistency-fine": 9,
        "remap-consistency-smallstep": 6,
        "remap-consistency-interval": 12,
        "material-minimax": 2,
        "minimax-return": 3,
        "diffusion": 20,
    }
    counts = [[name, len(raw[name]), count] for name, count in expected.items()]
    for first in range(0, len(counts), 25):
        r.table(["Campagne", "Enregistrés", "Prévus"], counts[first : first + 25])
    queue = args.input / "queue-status.json"
    if queue.exists():
        jobs = json.loads(queue.read_text())
        r.table(
            ["Exécution séquentielle", "État", "Code de sortie"],
            [[x["name"], x["status"], x.get("returncode", "en cours")] for x in jobs],
        )
    priority = args.input / "priority-status.json"
    if priority.exists():
        r.table(
            ["Reprise et diagnostics", "État", "Code de sortie"],
            [
                [x["name"], x["status"], x.get("returncode", "en cours")]
                for x in json.loads(priority.read_text())
            ],
        )
        r.paragraph(
            "Le premier lancement compressible a échoué pendant la capture CUDA du produit CSR CuPy. La reprise conserve les calculs sur GPU et utilise l'application creuse ordinaire. Les résultats réussis remplacent le fichier vide, tandis que le code de sortie initial reste dans l'historique."
        )
    for name in (
        "nested-sanitizer",
        "memory-transfer-sanitizer",
        "ambient-sanitizer",
        "chebyshev-sanitizer",
        "energy-mesh-sanitizer",
        "late-sanitizer",
        "coupled-sanitizer",
        "krylov-sanitizer",
        "reduction-sanitizer",
        "graph-sanitizer",
        "residual-energy-sanitizer",
    ):
        path = args.input / (name + ".log")
        if path.exists():
            lines = [
                line
                for line in path.read_text().splitlines()
                if "ERROR SUMMARY" in line
            ]
            r.paragraph(
                name
                + " : "
                + ("; ".join(lines) if lines else "contrôle encore incomplet")
            )
    r.paragraph(
        "Les validations indépendantes couvrent les degrés 1, 2 et 4, les projections, les adjoints en masse, les constantes, les changements de support et les identités de Gram. Toutes les opérations numériques — recouvrements, quadrature, réductions, matrices denses de contrôle — sont exécutées sur CUDA. Le processeur pilote les expériences et dessine les figures. Il s'agit de prototypes isolés dans experiments/slfe, pas d'une intégration de production au langage variationnel Rodin."
    )
    r.paragraph(
        "La piste d'article est le couplage FE avec mémoire sur supports variables, accompagné de garanties d'énergie et de coût mesuré. Les conditions manquantes sont une erreur contrôlée pour des vitesses FE générales, un problème couplé où la mémoire apporte un bénéfice indispensable, un contrôleur robuste pour plusieurs structures et une comparaison avec des concurrents GPU plus proches. Le résultat actuel ne justifie pas une revendication de rupture."
    )
    references = [
        (
            "Macías-Virgós, Pereira-Sáez et Tanré (2018), paramétrisations de Cayley sur les variétés de Stiefel",
            "https://doi.org/10.1016/j.geomphys.2017.08.011",
        ),
        (
            "Kim et Um (2026), reconstruction semi-lagrangienne presque orthogonale par noyau de Lanczos",
            "https://rtg.hufs.ac.kr/publications/SIGRAPHASIA26_lowres.pdf",
        ),
        (
            "Ouardghi, El-Amrani et Seaid (2022), enrichissement adaptatif semi-lagrangien FE",
            "https://doi.org/10.1016/j.compfluid.2022.105474",
        ),
        (
            "Keith et al. (2026), interpolation et optimisation pour le remap ALE FE",
            "https://arxiv.org/html/2611.05327v1",
        ),
        (
            "Plewa et Müller (1999), cohérence du transport des fractions de mélange",
            "https://arxiv.org/abs/astro-ph/9807241",
        ),
        (
            "Saad (1992), approximation de l'action exponentielle par Krylov",
            "https://epubs.siam.org/doi/10.1137/0729014",
        ),
        (
            "Simon et Behrens (2021), reconstruction semi-lagrangienne des sous-échelles",
            "https://doi.org/10.1007/s10915-021-01451-w",
        ),
        (
            "Arnold, Brezzi, Cockburn et Marini (2002), méthodes DG elliptiques",
            "https://www-users.cse.umn.edu/~arnold/papers/dgerr.pdf",
        ),
        (
            "Gaul, Gutknecht, Liesen et Nabben (2013), espaces de Krylov augmentés",
            "https://people.math.ethz.ch/~mhg/pub/defl-GGLN1-SIMAX.pdf",
        ),
        (
            "Codina, Principe, Guasch et Badia (2007), sous-échelles dynamiques",
            "https://doi.org/10.1016/j.cma.2007.01.002",
        ),
        (
            "Casas, Escorihuela-Tomàs et Moreno Casares (2025), formules de commutateurs",
            "https://doi.org/10.1007/s11128-025-04659-z",
        ),
        (
            "Adams, Finn, Knepley et Pusztay, rééchantillonnage par projection FE",
            "https://arxiv.org/abs/2501.13681",
        ),
        (
            "Golub et Varga (1961), semi-itérations de Chebyshev",
            "https://gdz.sub.uni-goettingen.de/download/pdf/PPN362160546_0003/PPN362160546_0003.pdf",
        ),
        (
            "Besse, Deriaz et Madaule (2017), adaptation multirésolution SL-DG",
            "https://www.oca.eu/images/LAGRANGE/pages_perso/nbesse/pub/NBesse_jcp17.pdf",
        ),
        (
            "Adams (2026), recouvrements SL-DG sur grilles adaptatives",
            "https://arxiv.org/abs/2603.19959",
        ),
        (
            "Nabizadeh et al. (2024), CO-FLIP et remise à zéro des cartes de flot",
            "https://cseweb.ucsd.edu/~ravir/sinasiga.pdf",
        ),
        ("Einkemmer et Moriggl, SL-DG sur GPU", "https://arxiv.org/abs/2212.03036"),
    ]
    for label, url in references:
        r.body.append(f"<p><a href='{url}'>{label}</a></p>")
        r.tex.append(r"\noindent\href{" + url + "}{" + tex_escape(label) + "}\\par")
    r.section("Reproduction")
    r.paragraph(
        "Les sources sont dans experiments/slfe : matched_benchmark.py, adaptive_study.py, moving_study.py, refinement_study.py, nested_validation.py et ambient_validation.py. Les fichiers results_night2 contiennent les paramètres, erreurs, historiques et empreintes du code. night_fields.py reproduit les champs; night_report.py reconstruit ce rapport sans simulation; restart_theory.tex fournit les hypothèses et preuves. Publication exclusivement sur la branche SLFE du fork keckj/rodin."
    )
    title = "SLFE : supports mobiles et transfert de mémoire"
    style = "body{max-width:1080px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#152c3a}h2{margin-top:2em}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ccd6df;padding:8px;text-align:left}img{max-width:100%}figcaption{font-size:14px}"
    (args.output / "rapport_nuit.html").write_text(
        "<!doctype html><html lang='fr'><meta charset='utf-8'><title>"
        + title
        + "</title><style>"
        + style
        + "</style><h1>"
        + title
        + "</h1>"
        + "\n".join(r.body)
        + "</html>"
    )
    preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}\usepackage[french]{babel}
\usepackage{amsmath,amssymb,geometry,graphicx,booktabs,hyperref}
\geometry{margin=2cm}\hypersetup{colorlinks=true,urlcolor=blue,hypertexnames=false}
\setlength{\emergencystretch}{3em}
\title{SLFE : supports mobiles\\et transfert de mémoire}
\author{Recherche GPU sur le fork keckj/rodin}\date{DATE}
\begin{document}\maketitle
""".replace(
        "DATE", tex_escape(stamp)
    )
    tex = preamble + "\n".join(r.tex) + r"\end{document}"
    for char, replacement in {
        "κ": r"\ensuremath{\kappa}",
        "γ": r"\ensuremath{\gamma}",
        "τ": r"\ensuremath{\tau}",
        "σ": r"\ensuremath{\sigma}",
        "10⁻¹¹": r"\(10^{-11}\)",
        "10⁻¹⁴": r"\(10^{-14}\)",
        "10⁻¹²": r"\(10^{-12}\)",
        "π": r"\(\pi\)",
        "≤": r"\(\leq\)",
        "−": "-",
        "²": r"\(^{2}\)",
        "×": r"\(\times\)",
    }.items():
        tex = tex.replace(char, replacement)
    (args.output / "rapport_nuit.tex").write_text(tex)
    with (args.output / "pdf-build.log").open("w") as log:
        for _ in range(2):
            subprocess.run(
                [
                    "lualatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "rapport_nuit.tex",
                ],
                cwd=args.output,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(
                generated=stamp,
                matched=summary,
                counts={k: len(v) for k, v in raw.items()},
            ),
            indent=2,
        )
        + "\n"
    )
    for path in args.output.glob("*.svg"):
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
