# Contrat normatif Omniverse photoréaliste pour données d'entraînement

## Statut et vocabulaire

Ce document fixe le résultat de production accepté pour la campagne
FireViewer Omniverse. Il s'applique au setup du pod, aux scènes USD, à la revue
humaine, aux simulations futures, aux observations produites et à leur
transfert.

- **MUST** indique une exigence obligatoire.
- **REFUS** indique qu'un seul écart invalide la scène, l'observation ou la
  campagne concernée.
- Un reçu automatique prouve uniquement les contrôles qu'il contient. Il ne
  prouve ni l'exécution sur RunPod, ni le rendu RTX, ni la qualité visuelle, ni
  la revue humaine.

Le code et les tests du dépôt peuvent préparer et contrôler ce contrat. Ils ne
doivent jamais déclarer qu'un pod, l'Editor, une simulation, un rendu ou un
transfert a réellement abouti sans l'artefact d'exécution correspondant.

## Portefeuille obligatoire

La campagne MUST contenir exactement :

- **4 scènes de base** explicitement fournies par l'opérateur ;
- **5 variantes fictives photoréalistes par base** ;
- **20 scènes finales**, ordonnées et identifiées `SIM-01` à `SIM-20`.

Chaque variante MUST conserver, par rapport à sa base :

- le nombre exact d'arbres et de bâtiments par famille ;
- tous les identifiants stables et numériques ;
- les références des assets USD ;
- les échelles individuelles ;
- l'identité et la topologie contrôlée des réseaux.

La forêt, les groupes bâtis et les routes MUST être réellement réarrangés de
façon déterministe et suffisamment différente entre les cinq variantes d'une
même base. Le réarrangement MUST rester compatible avec le relief, les
habitats, les fondations, les accès routiers, l'eau, les ponts, les distances
de sécurité et les contraintes physiques fixes. Une contrainte ne doit jamais
être relâchée en fonction du résultat à faire accepter.

## Qualité spatiale et visuelle

La qualité photoréaliste et l'utilité pour l'entraînement priment sur la
traçabilité documentaire. La provenance, les manifests et les hashes servent
uniquement à l'intégrité, à la reprise et au contrôle des artefacts : ils ne
peuvent jamais justifier un asset moins réaliste, une densité réduite, une zone
vide, une géométrie simplifiée ou un rendu médiocre.

Chaque scène MUST fournir sur toute son emprise :

- un terrain issu des sources LiDAR/MNT verrouillées, avec relief continu et
  matériau PBR sans objets incrustés ;
- une végétation complète et dense, instanciée depuis de vrais assets USD ;
- des bâtiments complets, rendables et ancrés au terrain ;
- des routes topologiques, ponts, surfaces d'eau et cours d'eau cohérents ;
  les routes visibles viennent du matériau de terrain dérivé de l'orthophoto,
  sans mesh ni asset routier dédié ;
- des payloads de terrain et d'objets tuilés ;
- des LOD distincts `HERO`, `MID` et `FAR` issus du même lineage d'asset.

Le terrain PBR MUST être object-free : aucune orthophoto brute ou texture visible ne
peut contenir des arbres, routes ou bâtiments qui deviendraient des images
fantômes après réarrangement. Les masques spatiaux peuvent uniquement piloter
le relief et le mélange des matériaux.

Pour les routes, les données d'orthophoto et les vecteurs source peuvent
piloter le matériau PBR du terrain. Leur rendu reste donc visible et cohérent
avec une variante, mais le pipeline ne crée ni ruban USD, ni matériau/asset
routier séparé. Les vecteurs restent obligatoires pour la topologie, le
placement des acteurs, les annotations et les contraintes de composition.

Chaque scène MUST exposer exactement 400 tuiles de terrain, 400 payloads
d'objets `HERO`, 400 `MID` et 400 `FAR`. Le terrain reste présent sur toute la
zone. Les objets visibles utilisent le niveau adapté à la caméra sans trou de
transition.

Sont des **REFUS** :

- cube, cône, cylindre, sphère, primitive générique ou placeholder utilisé à
  la place d'un asset final ;
- arbre, bâtiment, route, pont ou eau manquant ;
- tuile vide, zone vide créée par raccourci, rupture ou superposition de
  tuiles ;
- terrain plat, lissé, étiré, pixelisé ou portant de l'imagerie fantôme ;
- forêt anormalement clairsemée, amas d'arbres empilés ou famille incohérente
  avec son habitat ;
- bâtiment inventorié mais absent, flottant, empilé ou sans fondation viable ;
- route déconnectée, trop pentue, située dans l'eau sans pont, ou pont sans
  élévation vérifiée ;
- eau invisible, non matérialisée, dupliquée ou rendue comme une simple courbe
  sans surface ;
- changement de nombre, d'identité, d'asset ou d'échelle ;
- dépendance distante ou non verrouillée nécessaire à l'ouverture.

## Matériel et stockage du pod

Le pod de production MUST utiliser :

- **NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 Go** ;
- au moins `90000 MiB` de VRAM annoncés et utilisables ;
- au moins `138000 MiB` de RAM effectivement accessibles après prise en compte
  de la limite cgroup ;
- **1500 Go de NVMe éphémère** pour le conteneur.

Une autre famille de GPU est un **REFUS** pour cette campagne. Une limite
cgroup inférieure au seuil RAM est également un **REFUS**, même si l'interface
RunPod annonce davantage de mémoire.

La campagne MUST fonctionner sans volume persistant. Runtimes, caches,
sources, assets, scènes, frames et reçus restent sur le NVMe éphémère jusqu'au
transfert final. Le setup MUST contrôler l'espace libre et réserver la place
nécessaire avant chaque phase lourde.

Le pod MUST rester démarré jusqu'à un ordre explicite de l'opérateur. Aucun
script de préparation, de rendu, de transfert ou de reprise ne peut :

- arrêter ou supprimer automatiquement le pod ;
- supprimer automatiquement une source, une scène, une frame ou un reçu ;
- considérer l'arrêt du pod comme une étape normale de succès.

## Barrière humaine avant simulation

`prepare` MUST produire les quatre bases et les vingt variantes, puis s'arrêter
dans l'état `AWAITING_EDITOR_REVIEW`.

Le handoff dans l'Editor à l'opérateur n'est autorisé qu'après une QA interne
complète. Avant de solliciter sa validation, l'équipe de production MUST avoir
elle-même :

- accepté les gates structurels, géométriques, d'identité et de topologie ;
- accepté les 400 tuiles, les trois LOD, le PBR, les densités de végétation et
  le contenu bâti/routier/hydrologique ;
- accepté le gate de couverture et de non-occlusion des 40 caméras ;
- mesuré et accepté la stabilité RAM/VRAM sur le workflow de revue ;
- rendu un proof pack représentatif avec caméras verticale, basse et inclinée ;
- inspecté ce proof pack et enregistré une décision de QA interne positive
  liée par hash aux artefacts remis à l'opérateur.

Une ambiguïté, un défaut non expliqué, une image médiocre ou un échec de
stabilité bloque le handoff. La scène repart en correction et repasse tous les
gates touchés. La validation de l'opérateur est le dernier gate de qualité ;
elle ne doit jamais servir de séance de debug ou de détection initiale des
défauts.

Après cette QA interne seulement, la commande `review` MUST ouvrir la scène
finale complète `SIM-01` dans le véritable Omniverse Editor. La revue MUST
examiner au minimum :

- les 400 tuiles depuis des caméras verticales, basses et inclinées ;
- le relief, les coutures et les matériaux PBR ;
- la densité et la distribution de la végétation ;
- les bâtiments, routes, ponts et eaux ;
- les transitions `HERO`/`MID`/`FAR` ;
- la stabilité, la mémoire et la latence.

Avant un reçu humain d'acceptation de `SIM-01`, lié par SHA-256 au root USD, au
build, au runtime, aux assets et aux validations courantes :

- aucune simulation de feu ne peut commencer ;
- aucune frame de feu ne peut être produite ;
- aucun statut ne peut laisser entendre que le portefeuille est accepté.

Tout défaut visible est un **REFUS**. Un reçu `AUTO_VALIDATED` ou un test local
ne remplace jamais cette décision.

## Calendrier déterministe des vingt simulations

Après et seulement après l'acceptation humaine de `SIM-01`, la durée simulée
MUST suivre exactement ce tableau :

| Scène | Jours | Observations attendues |
| --- | ---: | ---: |
| `SIM-01` | 4 | 480 |
| `SIM-02` | 5 | 600 |
| `SIM-03` | 6 | 720 |
| `SIM-04` | 7 | 840 |
| `SIM-05` | 8 | 960 |
| `SIM-06` | 9 | 1 080 |
| `SIM-07` | 10 | 1 200 |
| `SIM-08` | 11 | 1 320 |
| `SIM-09` | 12 | 1 440 |
| `SIM-10` | 4 | 480 |
| `SIM-11` | 5 | 600 |
| `SIM-12` | 6 | 720 |
| `SIM-13` | 7 | 840 |
| `SIM-14` | 8 | 960 |
| `SIM-15` | 9 | 1 080 |
| `SIM-16` | 10 | 1 200 |
| `SIM-17` | 11 | 1 320 |
| `SIM-18` | 12 | 1 440 |
| `SIM-19` | 4 | 480 |
| `SIM-20` | 5 | 600 |
| **Total** | **153** | **18 360** |

Une journée MUST produire trois instants : **08:00**, **14:00** et **20:00**.
Chaque instant MUST être observé depuis exactement **40 viewpoints**, soit 120
observations par jour et exactement **18 360 observations** pour la campagne.

La distribution des durées et l'ordre `SIM-01..SIM-20` sont immuables. Une
durée choisie aléatoirement, un jour omis, un instant supplémentaire ou une
observation manquante est un **REFUS**.

## Contrat des 40 viewpoints

Chaque scène MUST disposer de 40 caméras de capture déterministes. Leurs poses
et intrinsics MUST rester strictement fixes pour tous les jours et les trois
instants de cette scène.

Avant le premier feu, un gate MUST prouver pour les 40 viewpoints :

- la couverture de l'emprise et des régions d'intérêt ;
- l'absence d'occlusion permanente rendant une vue inexploitable ;
- une altitude et une orientation valides, sans passage sous le terrain ;
- une projection et des intrinsics finis ;
- une identité stable `VIEW-01` à `VIEW-40`.

Une caméra peut différer entre deux scènes, mais elle ne peut pas dériver entre
les jours ou les heures d'une même scène. Une vue dupliquée, hors emprise,
entièrement occultée ou non reproductible est un **REFUS**.

## Séparation FireTruth / FireVisual et visibilité

Chaque stage de variante doit réserver deux racines USD distinctes :
`/World/FireTruth` et `/World/FireVisual`. Avant l'acceptation Editor, elles
restent vides et invisibles ; après l'acceptation, chaque état de feu les
compose ensemble avec le même identifiant d'état. `FireTruth` contient les
géométries et données de simulation servant aux annotations ; `FireVisual`
contient uniquement le rendu photoréaliste visible.

Le writer de capture FireViewer n'utilise pas `BasicWriter`. Pour chaque
observation, il doit publier séparément :

- le RGB rendu sous `FireVisual` ;
- la profondeur `distance_to_camera`, les segmentations sémantique et
  d'instance sous `FireTruth` ;
- un reçu de visibilité issu de raycasts PhysX closest-hit vers les cibles du
  front actif.

Une projection 2D seule ne vaut jamais preuve de visibilité. L'observation est
refusée si aucun rayon du front actif n'atteint la cible à la distance attendue
sans collision antérieure. Un changement de `fire_state_id` impose de créer un
writer neuf afin d'empêcher le mélange de vérité d'un instant avec les pixels
d'un autre.

## Contrat d'une observation

Chaque observation MUST contenir au minimum :

- une image RGB photoréaliste réellement rendue ;
- `simulation_id`, `base_scene_id`, `variant_id`, `day_index`,
  `view_id` et l'instant `08:00`, `14:00` ou `20:00` ;
- la pose caméra complète dans le repère local de la scène ;
- la pose ou transformation correspondante en `EPSG:2154` ;
- les intrinsics complets et la résolution de l'image ;
- l'heure simulée et son fuseau ;
- l'état du feu au même instant ;
- l'état météorologique utilisé au même instant ;
- le SHA-256 et la taille de chaque fichier ;
- les SHA-256 du root USD, du build, du contrat de caméra, de l'état feu et de
  l'état météo dont dépend la frame.

L'image et ses métadonnées MUST partager une clé d'observation canonique. Les
valeurs non finies, métadonnées absentes, poses incohérentes, états temporels
décalés ou fichiers non liés par hash sont des **REFUS**.

Un manifeste de campagne MUST inventorier exactement les 18 360 clés
d'observation attendues et reçues. Le compte ne peut pas être inféré du seul
nombre de fichiers, car une observation incomplète doit rester refusée.

## Reprise et atomicité

La production MUST être reprenable par frame :

1. une observation est écrite dans un emplacement temporaire dédié ;
2. l'image et les métadonnées sont contrôlées et hashées ;
3. le couple complet est publié atomiquement ;
4. le checkpoint est mis à jour seulement après cette publication.

Au redémarrage d'une phase, une frame existante ne peut être réutilisée
qu'après relecture de ses métadonnées, recalcul de ses SHA-256 et vérification
de toutes ses dépendances. Une frame partielle, périmée ou incohérente est
recalculée sans supprimer les autres frames valides.

Les builds des bases et des variantes MUST appliquer la même logique de reprise
par artefact hashé. Un crash ne doit pas imposer la reconstruction d'une scène
déjà intégralement vérifiée.

## Vingt exports autonomes

La sortie finale MUST contenir un export autonome par scène, soit vingt
packages distincts.

Chaque package MUST :

- se rouvrir dans un environnement isolé sans accès au pod d'origine ;
- n'utiliser que des chemins relatifs internes au package ;
- ne contenir aucune URL distante, aucun chemin absolu Linux ou Windows, aucun
  secret et aucune clé API ;
- inclure ou embarquer toutes les dépendances USD, matériaux, textures,
  payloads, métadonnées et manifests requis ;
- fournir un inventaire SHA-256 vérifiable avant et après transfert.

Un package qui dépend d'un cache du pod, d'un asset hors package ou d'un chemin
de la machine de production est un **REFUS**.

Les vingt packages et leurs observations MUST être transférés vers une
destination explicitement choisie sur le disque **D:**. Le transfert n'est
accepté qu'après comparaison de l'inventaire, des tailles et des SHA-256 à la
source. Ce reçu local n'autorise pas l'arrêt du pod : seul un ordre explicite
de l'opérateur le permet.

## Preuves obligatoires et frontière non vérifiée

Une campagne conforme MUST conserver au minimum :

- les reçus de prévol GPU, VRAM, RAM/cgroup, Vulkan et espace NVMe ;
- les locks et SHA-256 des sources LiDAR/MNT, assets USD et matériaux PBR ;
- les reçus des quatre builds de base ;
- le plan 4 × 5 et les reçus des vingt scènes ;
- les gates de tuiles, LOD, identités, nombres, topologie, matériaux et
  viewpoints ;
- le proof pack, ses mesures RAM/VRAM et la décision de QA interne positive ;
- le reçu de revue humaine de `SIM-01` ;
- les checkpoints et manifests des 18 360 observations ;
- les preuves de réouverture isolée des vingt exports ;
- le reçu de transfert et de vérification sur D:.

Tant que ces artefacts n'ont pas été produits par le workflow réel :

- le pod RunPod est **non vérifié** ;
- le build et l'ouverture de l'Editor sont **non vérifiés** ;
- la qualité photoréaliste est **non vérifiée** ;
- la revue humaine est **non effectuée** ;
- les simulations et 18 360 observations sont **non produites** ;
- les exports isolés et le transfert vers D: sont **non vérifiés**.

La présence du code, de tests locaux ou de contrats textuels ne doit jamais
être présentée comme la preuve de ces états d'exécution.
