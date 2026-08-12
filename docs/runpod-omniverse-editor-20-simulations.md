# Pod RunPod Omniverse Editor — portefeuille photoréaliste 4 × 5

Ce profil prépare un véritable FireViewer USD Composer, les sources
géospatiales LiDAR, une bibliothèque d’assets USD photoréalistes et les contrats
nécessaires à un portefeuille fixe de **20 scènes fictives** :

- exactement **4 scènes de base**, désignées explicitement par
  `FW_OMNI_BASE_ZONES` ;
- exactement **5 variantes cohérentes par base** ;
- donc exactement **20 scènes**, identifiées `SIM-01` à `SIM-20`.

Les variantes ne sont pas des reconstitutions fidèles des lieux réels. Le
terrain et l’eau de chaque base constituent le support spatial accepté, tandis
que la forêt, le bâti et les routes sont réarrangés de manière déterministe.
Chaque variante conserve les comptes par famille, les identifiants stables, les
références d’assets USD et leurs échelles. Elle doit rester physiquement
cohérente et visuellement photoréaliste.

Le setup s’arrête à `AWAITING_EDITOR_REVIEW`. Il ne lance ni simulation de feu
ni génération de données d’entraînement. L’ouverture de l’Editor est une
commande distincte, et une acceptation humaine liée aux artefacts courants est
obligatoire avant toute simulation future.

Le résultat obligatoire, les motifs de refus et le contrat post-acceptation
des 18 360 observations sont fixés dans le
[contrat normatif photoréaliste](omniverse-photoreal-training-contract.md).

## Ce que le code prépare, et ce qu’il ne prouve pas encore

| Préparé ou vérifié par le code | À exécuter et vérifier sur le pod |
| --- | --- |
| Contrat matériel RTX PRO 6000, mémoire GPU, RAM effective et pilote | Présence réelle du **RTX PRO 6000 Blackwell Server Edition 96 Go** choisi dans RunPod |
| Build reproductible de Kit App Template et de FireViewer USD Composer | Build Linux réel de Kit, Vulkan et ouverture de la fenêtre Editor |
| Catalogue externe validé et sélection explicite de quatre bases distinctes | Téléchargement complet des sources des quatre bases |
| Installation sûre et hashée du bundle USD/PBR, avec validation native prévue | Ouverture native réelle des USD, graphes UsdShade, textures et LOD avec le Python Isaac/Kit |
| Index fixe 4 × 5, seeds uniques et état `blocked_pending_editor_review` | Production native des quatre contrats de base, puis authoring des vingt variantes |
| Algorithme de composition et authoring USD tuilé HERO/MID/FAR | Inspection visuelle d’au moins une scène finale complète dans l’Editor |
| Gates structurels, géométriques et de cohérence | Qualité RTX, absence de défaut visible et performances réelles de l’Editor |

Un test local ou un reçu `AUTO_VALIDATED` ne prouve donc ni le rendu RTX, ni
l’absence de défaut visuel, ni l’acceptation humaine. Le reçu de revue pending
ne contient volontairement aucun champ `decision`.

## Contrat matériel du pod

Le profil de production est imposé :

- **GPU : NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 Go** ;
- seuil prévol GPU : au moins `90000 MiB` annoncés par `nvidia-smi` ;
- pilote NVIDIA : au moins `570.158.01` ;
- RAM configurée dans RunPod : **150 Go** ;
- seuil prévol RAM : au moins `138000 MiB` effectivement accessibles au
  conteneur ;
- stockage du conteneur : **1500 Go de NVMe éphémère**, sans volume
  persistant.

Le contrôle RAM ne se fie pas seulement à `/proc/meminfo`. Il retient la plus
petite limite finie entre la mémoire hôte annoncée et les limites cgroup v2
(`/sys/fs/cgroup/memory.max`) ou v1
(`/sys/fs/cgroup/memory/memory.limit_in_bytes`). Une allocation RunPod affichée
à 150 Go ne passe donc pas si le conteneur est réellement limité sous
`138000 MiB`.

Le script exige également Vulkan et refuse tout GPU dont le nom normalisé
n’est pas exactement `RTX PRO 6000 Blackwell Server Edition` (le préfixe
`NVIDIA` est toléré). Une A100, H100, L40/L40S, RTX A6000 ou RTX 6000 Ada
n’est pas un remplacement accepté pour ce profil.

Les runtimes, téléchargements, assets, scènes, frames et preuves restent sur le
NVMe éphémère sous :

```text
/workspace/fireviewer-omniverse
```

Le pod doit être créé avec 1500 Go. Le prévol exige initialement 300 Gio libres ;
les téléchargements, extractions d’assets, authorings et captures possèdent en
plus leurs propres limites et réserves. Les données lourdes ne doivent pas être
placées dans `/tmp` ou `/root`.

Cette campagne n’utilise aucun volume persistant. Elle dépend donc d’un
transfert final vérifié vers une destination explicitement choisie sur D:.
Aucun script ne supprime les données éphémères et le pod reste démarré jusqu’à
un ordre explicite de l’opérateur, y compris après un transfert réussi.

## Contrat des sources et des assets

Chaque base finale couvre exactement **400 tuiles de 1 km**. Avec
`FW_OMNI_LIDAR_SCOPE=full-zone`, les 400 sources LiDAR sont acquises et une
preuve vérifie notamment les classes sol, végétation et bâti. Le mode
`review-cameras` ne couvre qu’un sous-ensemble borné : il peut servir à un
diagnostic, mais ne peut pas être présenté comme la scène finale complète.

Le bundle d’assets doit rester entièrement local au volume et hashé. Il
contient :

- des assets arbres et bâtis USD réellement rendables ;
- pour chaque asset, trois représentations distinctes `HERO`, `MID` et `FAR`
  issues d’un lineage commun ;
- des matériaux PBR pour `forest_floor`, `grass`, `soil`, `rock`, `asphalt`,
  `gravel` et `water` ;
- des textures carrées d’au moins 2K pour couleur, normale et rugosité,
  connectées aux branches correspondantes d’un graphe UsdShade.

Les rôles `asphalt` et `gravel` sont des composantes du matériau de terrain
partagé, pilotées par les données spatiales/orthophoto. Ils ne justifient pas
la création de meshes, payloads ou assets routiers dédiés : la topologie des
routes est conservée séparément pour les contraintes et les acteurs.

Les validations natives refusent les primitives de remplacement, les
dépendances distantes, les fichiers sortant de l’inventaire verrouillé, les
LOD sans complexité décroissante et les matériaux sans géométrie rendable liée.

Le sol final utilise un matériau **PBR sans objets incrustés**. Une orthophoto
brute ne doit jamais devenir la texture visible du terrain : elle créerait des
arbres, routes ou bâtiments fantômes après réarrangement. Les données
spatiales peuvent piloter les hauteurs et poids de mélange, mais le layout
natif du portefeuille exige une surface `object_free_pbr` identifiée et
hashée. Une orthomosaïque, même nettoyée, ne remplace pas ce matériau final.

Le PBR doit suivre le découpage des payloads de terrain. Les textures de base
sont partagées, tandis que chaque tuile ne référence que ses propres masques
et champs de relief, recadrés sur son emprise verrouillée. Un shader unique qui
agrégerait les entrées spatiales des 400 tuiles est interdit : il annulerait le
bénéfice du streaming et rendrait la compilation ou l’évaluation RTX
imprévisible. Les UV métriques des matériaux restent continus entre tuiles ;
les masques locaux sont contrôlés séparément aux bordures.

## Contrat des vingt scènes fictives

Les quatre bases sont obligatoirement fournies par l’opérateur. Le setup ne
choisit pas automatiquement quatre zones du catalogue. Pour chaque base,
l’algorithme produit cinq variantes qui :

- gardent le terrain, le profil de l’eau et leurs empreintes validées ;
- conservent exactement les comptes d’arbres et de bâtiments par famille ;
- conservent les identifiants stables, références USD et échelles des assets ;
- redistribuent réellement la forêt sans amas pathologique, selon biome, sol,
  pente et distances aux routes, eaux et bâtiments ;
- déplacent les groupes bâtis sur des fondations viables et accessibles depuis
  le réseau routier ;
- réarrangent les routes tout en conservant connexité, drapage au terrain,
  pente admissible et ponts explicitement élevés au-dessus de l’eau ;
- restent mesurablement différentes des quatre autres variantes de la même
  base.

Aucune contrainte n’est relâchée silencieusement. Une composition impossible
échoue sans produire de portefeuille partiel.

Chaque scène authorée possède :

- 400 payloads de terrain ;
- 400 payloads d’objets `HERO` ;
- 400 payloads d’objets `MID` ;
- 400 payloads d’objets `FAR` ;
- soit 1 200 payloads d’objets tuilés, jamais un payload monolithique ;
- des collisions `NEAR` et `FAR`.

Dans l’Editor, toute tuile visible reçoit `HERO` ou `MID`; les autres gardent
`FAR`. Le plafond concerne uniquement `HERO`. Les promotions sont chargées et
stabilisées avant le déchargement de l’ancienne représentation. Le terrain
reste chargé sur toute la zone.

## Critère « aucune zone défectueuse »

Les gates automatiques refusent notamment :

- un catalogue incomplet ou une base dupliquée ;
- moins ou plus de 400 tuiles ;
- un payload terrain, `HERO`, `MID` ou `FAR` absent ;
- une tuile sans contenu déclaré ;
- un masque recadré sur l’emprise source globale au lieu de sa tuile ;
- un graphe de matériau global agrégeant les entrées des 400 tuiles ;
- un changement de compte, d’identifiant stable ou d’asset ;
- une surface avec imagerie d’objets fantômes ;
- un arbre hors habitat, dans l’eau, sur une route ou dans un bâtiment ;
- un bâti sans fondation viable ou sans accès routier ;
- une route déconnectée, trop pentue ou traversant l’eau sans pont déclaré ;
- un fichier lourd ou une dépendance USD sortant du volume verrouillé.

Ces contrôles ne remplacent pas l’inspection visuelle. La scène doit être
refusée si l’Editor montre une tuile vide, une rupture de terrain ou de
matériau, des objets empilés, une forêt clairsemée par erreur, des silhouettes
fantômes, un bâtiment flottant, une route incohérente, un cours d’eau défectueux
ou une latence empêchant la revue. **Un défaut visible signifie que la scène
n’est pas acceptée et qu’aucune simulation ne peut commencer.**

## Configuration

Copier les noms de variables de
`config/runpod-omniverse-editor.env.example` dans les secrets ou variables du
pod. Les valeurs obligatoires comprennent :

- `FW_ACCEPT_NVIDIA_EULA=YES`, après lecture des conditions NVIDIA ;
- `FW_OMNI_BASE_ZONES`, avec exactement quatre IDs distincts du catalogue ;
- `FW_OMNI_PILOT_ZONE`, appartenant à ces quatre bases, ou omission pour
  sélectionner la première ;
- soit `FW_SDG_ZONE_CATALOG_ROOT`, soit l’URL HTTPS du catalogue, son SHA-256
  et `FW_OMNI_CATALOG_ALLOWED_HOSTS` ;
- l’URL, le SHA-256 et l’allowlist HTTPS du bundle d’assets lorsque la
  bibliothèque NVIDIA disponible ne satisfait pas le contrat complet ;
- `FW_OMNI_REVIEW_PASSWORD`, secret d’au moins 16 caractères ;
- `FW_OMNI_LIDAR_SCOPE=full-zone` pour une validation finale.

## Préparation du pod et de la scène pilote

Depuis le checkout `fireviewer-sdg` présent sur le pod :

```bash
bash tools/runpod/setup-omniverse-pod.sh prepare
```

La commande est protégée par `flock` et reprend uniquement les phases déjà
prouvées :

1. installation des dépendances système ;
2. contrôle RTX PRO 6000, pilote, Vulkan, RAM effective et volume ;
3. environnement géospatial PDAL/GDAL verrouillé ;
4. clone du commit Kit App Template épinglé et build Linux de l’Editor ;
5. validation du catalogue et des quatre IDs de base explicites ;
6. runtime Isaac 6.0.1 persistant ;
7. installation et validation du bundle USD/PBR ;
8. index immuable des vingt emplacements `SIM-01` à `SIM-20` ;
9. acquisition LiDAR et des autres sources de la base pilote ;
10. build tuilé de la scène pilote ;
11. gate structurel/géométrique et reçu de revue pending.

La fin correcte est :

```text
AWAITING_EDITOR_REVIEW
No fire simulation has been started or authorized.
```

À ce stade, le code a construit la scène pilote et indexé le portefeuille. Il
n’a pas encore prouvé le téléchargement complet des trois autres bases, ni
authoré les vingt scènes finales.

Les phases peuvent être lancées séparément :

```bash
bash tools/runpod/setup-omniverse-pod.sh editor
bash tools/runpod/setup-omniverse-pod.sh catalog
bash tools/runpod/setup-omniverse-pod.sh assets
bash tools/runpod/setup-omniverse-pod.sh pilot
bash tools/runpod/setup-omniverse-pod.sh status
```

## Revue humaine obligatoire

La commande suivante est distincte de `prepare` :

```bash
bash tools/runpod/setup-omniverse-pod.sh review
```

Cette commande ne doit être remise à l’opérateur qu’après une QA interne
positive. Les gates structure, géométrie, identités, LOD, PBR, densité,
topologie et 40 caméras doivent être verts ; un test de stabilité RAM/VRAM et
un proof pack vertical/bas/incliné doivent aussi avoir été inspectés. Leur reçu
est lié par hash à `SIM-01`. Toute ambiguïté ou qualité médiocre bloque le
handoff et renvoie la scène en correction. La validation de l’opérateur est le
dernier gate, jamais une séance de debug.

RunPod Pods n’exposant pas UDP, la revue utilise un bureau distant HTTP/TCP
authentifié : `x11vnc` écoute seulement sur `127.0.0.1:5900`,
websockify/noVNC sur `127.0.0.1:6081`, puis nginx expose le port HTTP `6080`
avec Basic Auth. Ce transport montre la vraie fenêtre Kit/RTX ; ce n’est pas un
viewer WebGL de remplacement.

Dans l’Editor, vérifier au minimum :

- les 400 tuiles avec caméra verticale, basse et inclinée ;
- le relief, le PBR du terrain et l’absence d’imagerie fantôme ;
- la densité, la diversité et la répartition de la forêt ;
- le nombre, l’ancrage et la qualité des bâtiments ;
- les routes, ponts et cours d’eau ;
- les transitions `HERO`/`MID`/`FAR` et l’absence de trou visible ;
- la stabilité, la mémoire et la latence.

Après inspection réelle et seulement si aucun défaut n’est présent :

```bash
python3.12 -m fireviewer_sdg.omniverse_pod accept-review \
  --pending /workspace/fireviewer-omniverse/production/zone-scenes/<BASE>/editor-review-pending.json \
  --opened /workspace/fireviewer-omniverse/production/zone-scenes/<BASE>/review-opened.json \
  --output /workspace/fireviewer-omniverse/production/zone-scenes/<BASE>/editor-review-accepted.json \
  --reviewer "<identité du reviewer>" \
  --acknowledge "I inspected the scene in FireViewer USD Composer"
```

L’acceptation est liée par hash au runtime, au catalogue, aux assets, au build
et au root USD courants. Toute modification les invalide.

Cette commande ne lance toujours aucun feu. Le setup ne fournit aucun
entrypoint de simulation. Un futur consommateur devra passer
`assert_simulation_allowed` ou `simulation-gate` avec les artefacts et le reçu
d’acceptation actuels.

## Planification et authoring du portefeuille

Le module `fireviewer_sdg.native_variant_campaign` fournit deux opérations
séparées :

- `plan`, qui exige exactement quatre contrats de layout de base, un fichier de
  contraintes et un seed maître, puis produit atomiquement le plan 4 × 5 ;
- `author`, exécuté avec le Python Isaac/Kit, qui ouvre ce plan et écrit les
  vingt scènes USD tuilées.

Le setup `prepare` n’appelle pas automatiquement ces deux opérations. Avant de
les exécuter sur le pod, il reste donc à produire et valider les quatre layouts
de base, puis à choisir explicitement les chemins de plan et de sortie sous le
volume persistant. L’authoring conserve l’état
`blocked_pending_editor_review`; il ne lance aucune simulation de feu.

## Contrat post-acceptation, non exécuté par `prepare`

Une phase future de simulation ne devient autorisée qu’après le reçu humain
accepté de `SIM-01`. Elle doit conserver 40 viewpoints fixes par scène,
préalablement contrôlés pour la couverture, la non-occlusion, les intrinsics et
l’absence de passage sous terrain. Ces caméras ne bougent plus entre les jours
et les heures d’une même scène.

Les durées `SIM-01..SIM-20` sont exactement
`4,5,6,7,8,9,10,11,12,4,5,6,7,8,9,10,11,12,4,5` jours. Chaque journée possède
les trois instants `08:00`, `14:00` et `20:00`, avec les 40 viewpoints à chaque
instant. Le résultat attendu est donc exactement 153 jours et 18 360
observations.

Chaque observation associe atomiquement une image RGB réellement rendue à ses
identifiants scène/jour/heure/vue, à la pose locale et `EPSG:2154`, aux
intrinsics, à l’état feu et météo synchrone et aux SHA-256 de toutes ses
dépendances. Le manifeste énumère les 18 360 clés attendues. La reprise se fait
par frame après relecture des hashes ; une frame valide n’est ni écrasée ni
recalculée.

La livraison finale est composée de vingt exports autonomes réouvrables en
isolation, sans URL, chemin absolu, secret ou dépendance au pod. Leur transfert
vers D: doit être revalidé par inventaire, taille et SHA-256. Ni le transfert,
ni sa vérification ne donnent au workflow le droit de supprimer les données ou
d’arrêter le pod.

Le dépôt ne prouve actuellement ni l’exécution de cette phase future, ni le
rendu des 18 360 images, ni la réouverture isolée, ni le transfert vers D:.

## Données lourdes hors Git

Le catalogue reçu, les LiDAR, MNT/MNS/MNH, orthophotos, bundles d’assets,
runtimes, caches, USD authorés, rendus, résultats et archives restent sur le
NVMe éphémère du pod jusqu’au transfert final. Ils ne doivent pas être ajoutés
au dépôt.

Le `.gitignore` couvre notamment `assets/`, `downloads/`, `input/`,
`production/`, `renders/`, `results/`, `runtime/`, `workspace/`, le catalogue
`livrable_20_zones_france_omniverse/`, les formats USD, images, vidéos et
archives. Seuls le code, les contrats textuels, les exemples de configuration
sans secret et les tests appartiennent à Git.

## Références runtime

- [NVIDIA Kit App Template](https://github.com/NVIDIA-Omniverse/kit-app-template)
- [RunPod — Expose ports (TCP/HTTP, pas UDP)](https://docs.runpod.io/pods/configuration/expose-ports)
