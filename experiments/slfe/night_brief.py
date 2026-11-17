"""Four-page French research briefing from the recorded night experiments.

Distributed under the Boost Software License, Version 1.0.
Host-only rendering. The detailed report retains full data and limitations.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from report import Report, tex_escape, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = Report(args.output)
    stamp = datetime.now(ZoneInfo("Europe/Paris")).strftime("%d/%m/%Y, %H:%M CET")
    summary_path = args.output / "hamiltonian_summary.json"
    summary = (
        json.loads(summary_path.read_text())
        if summary_path.exists()
        else dict(matched=[])
    )

    def page(title):
        r.tex.append(r"\clearpage")
        r.section(title)

    def figure(name, caption, width=".98"):
        if not (args.output / (name + ".pdf")).exists():
            return
        r.body.append(
            f"<figure><img src='{name}.svg' alt='{caption}'><figcaption>{caption}</figcaption></figure>"
        )
        r.tex.append(
            r"\begin{center}\includegraphics[width="
            + width
            + r"\linewidth]{"
            + name
            + r".pdf}\\[3pt]{\small "
            + tex_escape(caption)
            + r"}\end{center}"
        )

    r.section("Une piste construite, une revendication à établir")
    r.paragraph(
        "Le travail a produit un couplage FE avec mémoire, des preuves conditionnelles d’énergie et d’erreur, et des prototypes CUDA vérifiés. Le choix du moment cible améliore fortement la compression d’une physique sous-résolue. L’extension du transport utilise une vitesse obtenue d’une fonction de courant FE par des cartes conservatives et des compositions temporelles d’ordre supérieur. Une avancée algorithmique publiable reste à démontrer face aux méthodes les plus proches; aucun résultat ne permet encore de qualifier l’ensemble de révolutionnaire."
    )
    figure(
        "night-method-schematic",
        "Le champ visible FE et sa mémoire physique sont transportés, soumis à la physique et transférés entre supports auxiliaires.",
        ".94",
    )
    r.paragraph(
        "Le résultat théorique central : tout transfert qui préserve l’observation FE et réduit l’énergie réduit aussi la distance à chaque référence FE. Cela reste vrai si l’espace de compression dépend de la solution. La borne globale dépend des défauts locaux de transport et de physique; elle n’ajoute pas un terme indépendant proportionnel au nombre de changements de support."
    )
    r.tex.append(
        r"\[\|Bq-u_h\|^2=\|q-u_h\|^2-\bigl(\|q\|^2-\|Bq\|^2\bigr),\qquad u_h\in V_h.\]"
    )
    r.paragraph(
        "Tous les calculs numériques, références, projections et contrôles matriciels utilisent la RTX 5090 en double précision. Les figures et la rédaction sont produites sur le CPU. Les sources restent des prototypes dans experiments/slfe, sur la branche SLFE du fork keckj/rodin."
    )
    r.body.append(
        "<p><a href='rapport_nuit.html'>Rapport détaillé illustré</a> · <a href='rapport_nuit.pdf'>PDF complet</a> · <a href='memory_paper_draft.pdf'>Manuscrit avec preuves</a></p>"
    )
    r.tex.append(
        r"\noindent\href{rapport_nuit.pdf}{Rapport détaillé illustré}\quad\href{memory_paper_draft.pdf}{Manuscrit avec preuves}\par"
    )

    page("Transport : ce que gagne la composition temporelle")
    r.paragraph(
        "Deux champs lisses sont transportés jusqu’à T = 0,5 par une vitesse à neuf directions de Fourier, issue de moments FE. Le comparateur DG/RK4 utilise le même espace P4 et la même vitesse. Les deux familles mettent à jour le champ FE à chaque pas et utilisent des graphes CUDA."
    )
    figure(
        "night-hamiltonian-cost",
        "Erreur relative maximale des deux champs contre temps GPU. Plein : mémoire persistante; pointillé : reconstruction FE à chaque pas.",
        ".98",
    )
    if summary["matched"]:
        labels = {
            "dg-central": "DG central",
            "dg-upwind": "DG upwind",
            "memory": "Mémoire / Y6",
            "reset": "Reset / Y6",
        }
        r.table(
            ["Méthode", "Pas", "Erreur relative", "Médiane (s)"],
            [
                [
                    labels[x["method"]],
                    x["steps"],
                    x["relative_error"],
                    x["median_seconds"],
                ]
                for x in summary["matched"]
            ],
        )
        dg = [
            x
            for x in summary["matched"]
            if x["method"].startswith("dg-") and x["relative_error"] <= 1e-7
        ]
        sl = [
            x
            for x in summary["matched"]
            if x["method"] == "memory" and x["relative_error"] <= 1e-7
        ]
        if dg and sl:
            best_dg = min(dg, key=lambda x: x["median_seconds"])
            best_sl = min(sl, key=lambda x: x["median_seconds"])
            r.paragraph(
                f"À la tolérance relative 1e−7 et n FE = 128, la mémoire/Yoshida est {best_dg['median_seconds']/best_sl['median_seconds']:.2f} fois plus rapide que le meilleur DG testé qui respecte cette tolérance. Les médianes proviennent de cinq blocs à ordre alterné; les étendues sont dans le rapport complet. Ce facteur concerne uniquement l’évolution après préparation, sur ce problème."
            )
    else:
        r.paragraph(
            "Les répétitions alternées ne sont pas encore toutes disponibles dans cet instantané. Le rapport détaillé conserve les mesures exploratoires et leurs paramètres."
        )
    r.paragraph(
        "Les compositions sont classiques; leur ordre continu ne constitue pas une nouveauté. La reconstruction FE peut être presque aussi efficace : le gain de transport ne démontre donc pas, à lui seul, l’utilité de la mémoire. Les assemblages, la préparation et toute physique supplémentaire sont exclus de ces chronométrages."
    )

    page("Mémoire physique : conserver directement le moment cible")
    r.paragraph(
        "Une correction algorithmique tardive améliore fortement le test de physique sous-résolue : l’espace de compression conserve maintenant le moment auxiliaire brut du champ cible, en plus de la mémoire entrante. Ce moment est déjà disponible. Deux directions remplacent ici les trois directions précédentes."
    )
    figure(
        "night-target-seed",
        "Une réaction crée des sous-échelles à partir d’un champ constant. L’erreur porte sur l’observation FE exacte de la solution physique, pas sur la représentation FE du champ continu.",
        ".98",
    )
    r.paragraph(
        "En arithmétique exacte, conserver le moment P_W f garantit que l’erreur de compression ne dépasse pas l’erreur de sa projection sur W. Les observations FE, la contraction et la mémoire entrante restent compatibles. Le seuil numérique de rang doit toutefois conserver les directions nécessaires; les tests exposent une limite de conditionnement près des dépendances linéaires."
    )
    unresolved = [x["result"] for x in load(args.input, "unresolved-reaction")]
    unresolved += [
        x["result"]
        for x in load(args.input, "unresolved-target")
        if x["result"]["method"] == "target"
    ]
    labels = {
        "full": "Ancienne base",
        "target": "Moment cible retenu",
        "reset": "Reconstruction FE",
        "auxiliary": "Auxiliaire direct",
    }
    selected = [
        x for x in unresolved if x["degree"] == 1 and x["nfe"] == 4 and x["steps"] == 64
    ]
    if selected:
        r.table(
            ["Méthode", "Erreur observation", "Évolution (s)"],
            [
                [
                    labels[x["method"]],
                    x["relative_observation_error"],
                    x["evolution_seconds"],
                ]
                for x in selected
            ],
        )
    r.paragraph(
        "Cas du tableau : réaction seule, P1 sur le maillage grossier n = 4, support auxiliaire Q4 de résolution 128, 64 pas. Les temps sont des mesures uniques hors préparation. Le solveur auxiliaire direct reste beaucoup moins cher : ce test prouve l’intérêt du choix de direction pour la compression, sans établir un avantage global de coût."
    )
    r.paragraph(
        "Sur les advections-réactions résolues, la reconstruction FE reste un concurrent sérieux. La réaction cubique précédente donne une meilleure précision avec mémoire, mais à un coût nettement supérieur. Les résultats défavorables et les contrôles indépendants restent dans le rapport complet."
    )

    page("Portée de la vitesse FE et décision pour l’article")
    r.paragraph(
        "Le second test utilise une fonction de courant à spectre infini. Les caractéristiques de référence suivent sa vitesse analytique non tronquée; le rayon spectral des moments FE est raffiné séparément. Cela vérifie la chaîne FE → vitesse filtrée → transport au-delà du cas à neuf modes."
    )
    figure(
        "night-hamiltonian-broad",
        "Raffinement du filtrage et coût de la décomposition. La convergence en bande spectrale ne signifie pas que cette approche soit rentable pour toute vitesse FE.",
        ".98",
    )
    broad = [x["result"] for x in load(args.input, "hamiltonian-broad-tail")]
    selected = [
        x
        for x in broad
        if x["nfe"] == 128
        and x["band"] == 10
        and x["steps"] == 8
        and x["method"] == "memory"
    ]
    if selected:
        x = selected[0]
        r.paragraph(
            f"Avec 220 vecteurs d’onde retenus, huit pas et n FE = 128, l’erreur relative est {max(x['relative_l2']):.3g}, pour un plancher FE de {max(x['relative_fe_floor']):.3g}. La vitesse filtrée diffère de la vitesse analytique de {x['velocity_relative_probe_error']:.3g} en norme relative échantillonnée. Le prix des nombreux cisaillements reste une limite importante."
        )
    r.paragraph(
        "La piste défendable pour un article est la compatibilité entre mémoire physique, observation FE, compression dépendant de l’état et physique dissipative. Les briques de projection, de sous-échelles, de cisaillement et de composition ont des antécédents : les preuves seules n’établissent pas la nouveauté bibliographique."
    )
    r.paragraph(
        "Les prochaines expériences décisives doivent établir un bénéfice à précision et coût comparables dans un problème couplé où la mémoire est nécessaire, contrôler les défauts pour des vitesses FE générales, puis comparer avec les remaps et méthodes de sous-échelles GPU les plus proches. Le prototype périodique 2D ne couvre encore ni frontières physiques, ni MPI, ni intégration de production dans Rodin."
    )
    r.paragraph(
        "Limite établie : les sous-pas négatifs restent contractants mais amplifient la dissipation des projections à maillage fixé. Les ordres quatre et six mesurés avant le plateau spatial ne doivent pas être annoncés comme un théorème d’ordre uniforme."
    )

    style = "body{max-width:1040px;margin:40px auto;padding:0 24px;font:17px/1.55 system-ui;color:#183345}h1{font-size:36px}h2{margin-top:2em}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border-bottom:1px solid #ccd6df;padding:7px;text-align:left}figcaption{font-size:14px;color:#456}a{color:#17658b}"
    (args.output / "synthese_08h.html").write_text(
        "<!doctype html><html lang='fr'><meta charset='utf-8'><title>SLFE — synthèse de recherche</title><style>"
        + style
        + "</style><h1>SLFE — synthèse de recherche</h1><p>"
        + stamp
        + "</p>"
        + "\n".join(r.body)
        + "</html>"
    )
    preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}\usepackage[french]{babel}
\usepackage{amsmath,amssymb,graphicx,geometry,booktabs,hyperref}
\geometry{margin=1.8cm}\hypersetup{colorlinks=true,urlcolor=blue}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}
\begin{document}
{\Large\bfseries SLFE : synthèse de recherche}\hfill STAMP\par
""".replace(
        "STAMP", tex_escape(stamp)
    )
    tex = preamble + "\n".join(r.tex) + r"\end{document}"
    for a, b in {
        "−": "-",
        "→": r"\(\to\)",
        "²": r"\(^{2}\)",
        "×": r"\(\times\)",
    }.items():
        tex = tex.replace(a, b)
    (args.output / "synthese_08h.tex").write_text(tex)
    with (args.output / "synthese_08h-build.log").open("w") as log:
        subprocess.run(
            [
                "lualatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "synthese_08h.tex",
            ],
            cwd=args.output,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


if __name__ == "__main__":
    main()
