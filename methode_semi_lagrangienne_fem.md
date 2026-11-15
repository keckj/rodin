# Méthode semi-lagrangienne remaillée sur éléments finis

Note de cadrage — 15 novembre 2026

## 1. Intention

Développer une méthode de transport semi-lagrangienne sur maillages éléments finis non structurés, en prolongeant l'esprit des méthodes particulaires remaillées d'ordre élevé de Jean-Baptiste Keck : transport suivant les caractéristiques, puis transfert sur une représentation spatiale maîtrisée.

L'objectif est de combiner une bonne précision d'advection, la conservation et un coût de calcul adapté aux GPU et au calcul distribué. Le cœur de la méthode doit pouvoir être couplé à différents solveurs physiques.

Les orientations retenues sont :

- Un transport **forward** comme point de départ privilégié, cohérent avec les méthodes remaillées existantes.
- Un transfert dans un espace FE, avec une cible d'ordre élevé et une conservation à établir au niveau discret.
- Une **positivité non imposée** : les noyaux de remaillage d'ordre élevé existants ont des poids négatifs. Des dépassements sont possibles et doivent être mesurés.
- Une fonction distance éventuellement utilisée comme outil géométrique pour les frontières.
- Une architecture permettant de comparer assemblage classique, assemblage partiel et matrix-free.

Le choix entre éléments continus et discontinus, les ordres, la quadrature et la stratégie exacte de transfert restent ouverts. La formulation ci-dessous constitue une base de travail à analyser et à valider.

## 2. Problème modèle

Commencer par le transport d'un scalaire conservatif, avec une vitesse donnée :

$$
\partial_t q+\nabla\cdot(\mathbf u q)
-\nabla\cdot(D\nabla q)=s(q,\mathbf x,t).
$$

Le développement doit d'abord isoler l'advection pure, puis introduire diffusion, sources et réactions. Le transport de plusieurs champs par une même vitesse permettra de mutualiser les trajectoires et les recherches géométriques.

La formulation conservative doit être distinguée de l'équation advective :

$$
\partial_t q+\mathbf u\cdot\nabla q=0.
$$

Ces deux équations coïncident lorsque la vitesse est à divergence nulle. Cette propriété n'est pas automatiquement satisfaite point par point par une vitesse FE issue d'un solveur incompressible.

## 3. Base proposée : transport forward et projection FE

### 3.1 Flot et transfert faible

Soit $F_n$ le flot de $t^n$ à $t^{n+1}$, obtenu en intégrant :

$$
\dot{\mathbf X}(t)=\mathbf u(\mathbf X(t),t),
\qquad \mathbf X(t^n)=\mathbf a.
$$

Pour l'advection conservative, une base naturelle est :

$$
\int_{\Omega^{n+1}}q_h^{\mathrm{adv}}(\mathbf x)v_h(\mathbf x)\,d\mathbf x
=\int_{\Omega^n}q_h^n(\mathbf a)v_h(F_n(\mathbf a))\,d\mathbf a.
$$

Cette identité est écrite ici pour un domaine transporté par le flot, ou un domaine fixe invariant, sans échange au bord. Les domaines ouverts demandent un traitement explicite des entrées et sorties.

Avec les fonctions de base d'arrivée $\varphi_i^{n+1}$, elle donne :

$$
M^{n+1}\mathbf c^{\mathrm{adv}}=\mathbf b^{\mathrm{adv}},
\qquad
b_i^{\mathrm{adv}}=\int_{\Omega^n}q_h^n(\mathbf a)
\varphi_i^{n+1}(F_n(\mathbf a))\,d\mathbf a.
$$

Le transport intervient dans le second membre. Il n'est pas nécessaire de construire une matrice globale de transfert entre les deux espaces FE.

### 3.2 Réalisation par points de quadrature

Une réalisation possible consiste à :

1. Évaluer le champ sur des points de quadrature de chaque élément de départ.
2. Transporter ces points par intégration des caractéristiques.
3. Localiser leurs positions dans le maillage d'arrivée.
4. Accumuler leurs contributions sur les fonctions de base d'arrivée :

$$
b_i^{\mathrm{adv}}\simeq
\sum_{K,r}w_{K,r}\,q_h^n(\mathbf a_{K,r})
\varphi_i^{n+1}(F_n(\mathbf a_{K,r})).
$$

Les poids $w_{K,r}$ incluent le volume physique. Cette procédure est un analogue FE du transport suivi d'un dépôt sur grille.

En FE continus, la projection avec masse consistante implique un solveur global. En DG, la matrice de masse est locale par élément. Une masse condensée est une variante possible, dont les effets sur la précision doivent être évalués.

### 3.3 Conservation et ordre

Si la constante appartient à l'espace test, prendre $v_h=1$ fournit le bilan global de masse. Sa réalisation numérique exige des intégrations cohérentes et un traitement correct des flux de frontière.

La précision dépend simultanément de l'espace FE, des trajectoires et du transfert. Une quadrature usuelle sur les éléments de départ peut sous-intégrer les fonctions de base d'arrivée composées avec le flot, notamment lorsque les points traversent plusieurs éléments. Quadrature adaptative, subdivisions ou intersections de maillages sont des pistes à comparer.

Les noyaux cartésiens existants ne se transposent pas directement aux simplexes. Il faut caractériser la reproduction polynomiale, les moments du transfert et l'erreur introduite par la projection. Les moments physiques d'une solution transportée ne sont pas tous invariants sous un flot général.

La conservation ne garantit ni la positivité ni la stabilité. Un éventuel limiteur serait une extension distincte, et non un prérequis de cette méthode.

## 4. Couplage aux opérateurs FE et gains recherchés

Sur maillage fixe, un premier prototype advection–diffusion peut effectuer le transport puis une diffusion implicite :

$$
(M+\Delta t K)\mathbf c^{n+1}
=\mathbf b^{\mathrm{adv}}+\Delta t\,\mathbf f^{n+1},
$$

où $K$ contient la diffusivité. Cette écriture est un point de départ au premier ordre ; une montée en ordre doit traiter de façon cohérente trajectoires, diffusion, sources et découplage.

Les gains à rechercher et à mesurer sont :

- **Précision du transport** : limiter la diffusion numérique et les erreurs accumulées lors des transferts.
- **Pas de temps** : permettre des déplacements traversant plusieurs mailles sans subir la CFL d'un transport eulérien explicite. La précision des trajectoires, la déformation du flot et les autres physiques restent contraignantes.
- **Assemblage** : retirer la convection du système implicite et construire le transport directement au second membre.
- **Réutilisation** : conserver les matrices et éventuellement leur factorisation lorsque maillage, pas de temps, coefficients et opérateurs de frontière sont fixes.

Face à un schéma déjà implicite, le bénéfice doit être évalué à précision comparable. Des sources, une viscosité ou des réactions variables peuvent encore modifier la matrice. Un découplage des réactions en problèmes locaux est une option à étudier.

## 5. Géométrie, frontières et maillage mobile

### Fonction distance

Une distance signée peut servir à détecter une rencontre avec la frontière le long d'une trajectoire, à estimer une normale et, dans une zone où la projection est unique, à retrouver un point proche sur la surface.

Elle n'est pas nécessaire au transport dans le volume et ne remplace ni la localisation des éléments ni l'identification des différents types de frontière. L'imposition d'une condition physique reste propre au problème : donnée d'entrée, flux, paroi mobile, etc. Arrêter ou projeter arbitrairement une trajectoire sur le bord ne suffit pas à définir cette condition.

Il n'est donc pas nécessaire de faire du level set le cœur de la méthode.

### Déplacement et remaillage

Il faut distinguer le déplacement temporaire des points transportés du déplacement du maillage support des FE.

| Situation | Conséquence pour les opérateurs |
| --- | --- |
| Transport puis retour sur le même maillage fixe | Pas de mise à jour géométrique de $M$ et $K$. |
| Déformation du maillage à connectivité fixe | Mise à jour des volumes, gradients et coefficients ; structure creuse généralement réutilisable. |
| Changement de connectivité | Nouveau transfert, mise à jour des degrés de liberté et reconstruction des structures concernées. |

Une factorisation numérique n'est généralement plus exacte lorsque les coefficients changent. Revenir à un domaine de référence ne rend pas les opérateurs constants : les Jacobiennes et métriques dépendent alors du temps.

Une extension ALE doit tenir compte du mouvement de la grille ; la convection relative fait intervenir $\mathbf u-\mathbf w$, où $\mathbf w$ est la vitesse du maillage. Les bilans et la cohérence géométrique doivent être vérifiés séparément du transport physique.

## 6. Architecture de calcul

Prévoir des composants distincts pour l'intégration des trajectoires, la localisation, le traitement des frontières, le transfert FE et l'application des opérateurs physiques.

L'application matrix-free des opérateurs est une option indépendante du SL : un transport eulérien peut également être matrix-free. Elle consiste à fournir au solveur l'action $\mathbf y=A\mathbf x$ par calcul élémentaire sans stocker la matrice globale.

- En assemblage partiel, stocker les facteurs géométriques et physiques utiles aux applications répétées.
- En matrix-free complet, recalculer ces facteurs pendant l'application.
- Comparer avec une matrice assemblée, particulièrement pour les éléments de bas ordre.
- Prévoir un préconditionneur efficace, éventuellement construit à partir d'une matrice approchée et réutilisée.

En MPI, les échanges de valeurs fantômes, les sommes aux degrés de liberté partagés et les réductions des solveurs restent présents. Le transport SL ajoute la gestion des trajectoires ou dépôts traversant les partitions. Un grand pas peut donc augmenter le coût des communications et de la localisation.

L'objectif est de réduire le temps total à précision fixée, en mesurant séparément transport, transfert, application des opérateurs, préconditionnement et communications.

## 7. Développement et validation proposés

1. **Advection sur domaine fixe périodique** : vitesse imposée, prototype conservatif, comparaison de plusieurs transferts et ordres FE.
2. **Advection–diffusion et sources** : couplage aux opérateurs FE et mesure des gains de réutilisation.
3. **Frontières physiques** : cas d'entrée/sortie, géométries courbes et contribution éventuelle d'une distance signée.
4. **Maillage mobile et remaillage** : transfert entre espaces successifs et contrôle des bilans géométriques.
5. **GPU et MPI** : optimisation à partir des coûts observés, comparaison assemblé/partiel/matrix-free.

Les cas tests doivent inclure translation, rotation, déformation réversible et transport compressible avec bilan connu. Mesurer les erreurs $L^1$, $L^2$ et $L^\infty$, la convergence spatiale et temporelle, le bilan de masse incluant flux et sources, ainsi que l'amplitude des sous-dépassements et sur-dépassements.

Les comparaisons de performance doivent porter sur une erreur cible commune. Vérifier aussi le comportement lorsque le pas diminue : multiplier les transferts peut accumuler leur erreur.

## 8. Positionnement et choix ouverts

Le semi-lagrangien FE, les méthodes Lagrange–Galerkin et la réutilisation des matrices sont établis. Une éventuelle contribution devra porter sur une construction précise : qualité du transfert d'ordre élevé, conservation, robustesse sur simplexes, frontières, maillages mobiles ou efficacité GPU/MPI. La nouveauté de cette combinaison reste à examiner.

Les principaux choix à arbitrer sont :

- FE continus ou DG ; ordre polynomial et représentation du champ.
- Projection consistante, masse condensée ou autre transfert conservatif.
- Stratégie de quadrature et coût acceptable de la localisation.
- Ordre temporel et découplage des opérateurs.
- Traitement des frontières et place de la fonction distance.
- Assemblage, préconditionnement et organisation parallèle.

## Références de départ

- [Keck et al., *Double-diffusive sedimentation at high Schmidt numbers: Semi-Lagrangian simulations*, 2021](https://doi.org/10.1103/PhysRevFluids.6.L022301) — contexte des méthodes particulaires remaillées.
- [Douglas et Russell, *Numerical Methods for Convection-Dominated Diffusion Problems Based on Combining the Method of Characteristics with Finite Element or Finite Difference Procedures*, 1982](https://doi.org/10.1137/0719063) — fondements caractéristiques–FE.
- [Colera, Carpio et Bermejo, *A nearly-conservative, high-order, forward Lagrange–Galerkin method for the resolution of scalar hyperbolic conservation laws*, 2021](https://doi.org/10.1016/j.cma.2020.113654) — formulation forward à comparer.
- [Cai, Guo et Qiu, *A high order conservative semi-Lagrangian discontinuous Galerkin method for two-dimensional transport simulations*](https://arxiv.org/abs/1612.06977) — voie SL-DG conservative ; prépublication 2016, version révisée 2017.
- [FreeFEM — Navier–Stokes](https://doc.freefem.org/models/navier-stokes-equations.html) — traitement par caractéristiques et réutilisation des matrices.
- [MFEM — Niveaux d'assemblage](https://mfem.org/howto/assembly_levels/) — assemblage complet, élémentaire, partiel et matrix-free.

