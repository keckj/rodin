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
            "source-smoke",
            "source-full",
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
            "moving-resolution",
            "moving-fields",
        )
    }
    r.paragraph(
        "Résultats disponibles le "
        + stamp
        + " (Europe/Paris). Calculs en double précision sur une RTX 5090. Les figures représentent les solutions calculées, avec une référence indépendante par caractéristiques."
    )
    r.section("Résultat et portée")
    r.paragraph(
        "Un support auxiliaire mobile résout le défaut observé lorsque la bosse quitte la zone fine d'une grille fixe. Le couplage FE avec mémoire accepte désormais un changement de support contractant, et un raffinement exactement emboîté sans perte d'énergie. Les formules s'exécutent sans racine ni inversion de matrice. Ces propriétés sont démontrées dans restart_theory.pdf et vérifiées sur GPU; elles n'établissent ni une nouveauté bibliographique, ni une convergence générale vers l'EDP."
    )
    r.paragraph(
        "Deux méthodes sont distinguées : le couplage avec mémoire maintient le champ FE à chaque changement; la référence d'advection pure transporte uniquement le champ DG auxiliaire entre une projection FE initiale et une projection FE finale. Cette seconde méthode est la plus rapide ici, mais elle n'offre pas les opérations physiques FE intermédiaires du couplage."
    )
    r.paragraph(
        "Les contre-tests exposent deux limites importantes : un transfert itératif tronqué peut dissiper même à support inchangé, et un contrôle de maillage tous les quatre pas modifie le problème d'adaptation lorsque le pas diminue. Une variante corrige le premier défaut au prix d'une non-linéarité; une ablation à intervalle physique fixé isole le second. La composition hamiltonienne par SVD perd la comparaison de coût et de précision contre Fourier/RK4 sur le vortex résolu. Ces résultats orientent la suite de la recherche."
    )
    for path in (args.input / "figures").glob("*"):
        if path.suffix in (".svg", ".pdf", ".png", ".json"):
            shutil.copy2(path, args.output / path.name)
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
            "Le flot autonome est suivi pendant une durée T, puis sa vitesse est inversée pendant la même durée. La solution exacte finale est le profil initial : aucune intégration de caractéristiques n'est nécessaire pour cette référence. Les partitions restent adaptées au champ calculé pendant les deux trajets. La dissipation et les pertes de mémoire ne s'annulent pas lors du retour."
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
    material = [x["result"] for x in raw["material-monitor"] + raw["material-phase"]]
    if material:
        r.section("La coupure périodique du moniteur déplace les quantiles")
        r.paragraph(
            "Des poids d'énergie positifs sont portés par des points matériels suivis par les caractéristiques. L'incompressibilité conserve ces poids. Ce diagnostic n'utilise aucune évolution FE/DG ni aucun remaillage du champ. Les grands déplacements des partitions sont reproduits à 512 et 1024 points par direction. Décaler la coupure périodique réduit le mouvement des quantiles diagonaux : une part importante du phénomène est géométrique, et non une instabilité de projection."
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, ax in zip((2, 3), axes):
            for x in material:
                label = (
                    "Phase ajustée"
                    if x.get("floating_phase")
                    else ("Coupe décalée" if x["shifted"] else "Coupe fixe en zéro")
                )
                ax.plot(
                    [h["time"] for h in x["history"]],
                    [h["axis_displacement"][axis] for h in x["history"]],
                    "--" if x["samples"] == 512 else "-",
                    label=f"{label}, {x['samples']}",
                )
            ax.axhline(0.01, color="black", linestyle=":")
            ax.set(
                title="Coordonnée x+y" if axis == 2 else "Coordonnée x−y",
                xlabel="Temps physique",
                ylabel="Déplacement maximal des arêtes",
            )
            ax.legend(fontsize=7)
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
    r.section("Contrôles et limites pour un article")
    r.paragraph(
        "La note établit également une borne d'erreur pour des supports variables : une transition contractante qui envoie l'état minimal J_n v sur J_(n+1) v n'ajoute pas de terme explicite dépendant du nombre de remaillages. Les défauts locaux de transport restent dans la borne, notamment le commutateur entre projection et générateur. Ce résultat conditionnel ne fournit ni un ordre de convergence pour les noyaux actuels ni une garantie d'efficacité du contrôleur."
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
        "source-full": 16,
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
    }
    r.table(
        ["Campagne", "Enregistrés", "Prévus"],
        [[name, len(raw[name]), count] for name, count in expected.items()],
    )
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
