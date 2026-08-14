# FireViewer — pilote Omniverse real-to-sim / sim-to-real

Worker GPU headless destiné à créer **de nouveaux cas** FireViewer à partir de captures NVIDIA
NuRec réelles ou de scènes Omniverse USD inspirées de terrains français. Il ne
copie, ne réétiquette et ne complète aucun corpus existant. Chaque provenance reste explicite et
une scène synthétique ne peut jamais être présentée comme un lieu réel.

Le pilote suit deux directions distinctes :

1. **real-to-sim** : orthophoto et MNT français, géométrie USD métrique, végétation et bâti
   SimReady verrouillés, puis ajout contrôlé du feu, de la fumée et des acteurs ;
2. **sim-to-real** : les pilotes acceptés ne servent au training qu'après comparaison avec des
   photos réelles françaises tenues à l'écart. Le pipeline ne déclare donc pas le domaine gap
   résolu simplement parce qu'un rendu paraît photoréaliste.

## Pipeline NVIDIA verrouillé

Le chemin visuel accepte deux provenances explicitement séparées : une capture NuRec réelle, ou
une nouvelle scène de référence synthétique construite sur des sources géographiques françaises.
Le chemin NuRec suit la documentation NVIDIA NuRec/NCore/3DGRUT et Isaac Sim :

1. nouvelle capture terrestre d’au moins 100 images 4K chevauchantes ;
2. calibration NCore v4 ou COLMAP, avec au moins 95 % d’images enregistrées et une erreur moyenne
   de reprojection inférieure ou égale à 1 px ;
3. reconstruction `nv-tlabs/3dgrut` ou NVIDIA NRE exportée en ParticleField USD (préféré) ou NuRec
   USDZ ;
4. évaluation sur au moins 10 vues tenues à l’écart, avec PSNR >= 25 et SSIM >= 0,90 ;
5. référence de la reconstruction dans un stage USD racine inscriptible et payload Flow ; les
   acteurs restent facultatifs et ne sont requis que lorsqu'une campagne active
   `response_engagement` ;
6. pour un vrai actif NuRec uniquement, `isaacsim.replicator.nurec_utils.setup_for_rendering(stage)`
   avant la première synchronisation Hydra ; les USD SimReady inspirés du réel n'importent pas
   ce module expérimental ; un seul feu par processus Isaac, stage courant inscriptible déjà attaché
   par `SimulationApp` sans réouverture du contexte, références USD/Flow stabilisées timeline
   arrêtée, puis chauffe Flow avant la création du graphe Replicator ; rendu mono-GPU RTX à
   1 280 x 720 pour le profil local Windows ou 1 920 x 1 080 pour les lots HD, 16 sous-trames
   et 48 pas de chauffe Flow ; ACES, éclairage extérieur
   calibré et validation d'histogramme avec reprise d'exposition empêchent l'enregistrement
   d'une image brûlée, noire ou sans dynamique ;
7. génération d’un cas, inspection humaine dans la console, puis seulement reprise de la
   production.

Avec `FW_SDG_PREPARE_IGN_CATALOG=1`, le bootstrap prépare le catalogue pilote
`/workspace/fireviewer-sdg/input/event-catalog-4096-hd-v2.json`. Il télécharge pour **trois sites
pilotes** ruraux, montagneux et agricoles une orthophoto IGN 4 096² et un MNT LiDAR-HD 2 048² sur
la même emprise EPSG:2154, puis construit un maillage USD 257² de 2 km de côté. Ces trois sites
prouvent le setup ; ils ne constituent pas la diversité géographique finale.

Si aucun manifeste n'est fourni, le worker inventorie avec `omni.client` les racines publiques
épinglées d'Isaac Sim 6.0, sélectionne au moins six végétations et un bâti rural, puis écrit des
wrappers USDA locaux et un lockfile contenant URI, hash/version fournisseur, taille, licence et
état de revue. Le cache de résolution reste sous `/opt/fireviewer-cache`, pas dans les livrables.
Les assets communautaires ou propriétaires ne sont acceptés que via
`FW_SDG_SIMREADY_ASSET_MANIFEST` après revue de leur licence et de leur provenance.

Le décor est techniquement préparé avant Flow. Les classes SDIS, Canadair, Dash et hélicoptère de
Sécurité civile exigent une correspondance nominale stricte : un camion ou hélicoptère générique
ne satisfait jamais la classe. Ces classes, les personnes et la catégorie
`response_engagement` sont explicitement hors périmètre de la V1 Windows locale. Leur absence ne
bloque donc pas ses trois catégories actives. Une future campagne qui réactive
`response_engagement` reste bloquée si l'une des sept classes acteur manque. Après satisfaction
des gates actifs, le feu, les caméras, la végétation et le bâti sont
recalés sur le même MNT. Le catalogue
décrit au moins 512 feux distincts, de durée variable entre 1 et 15 jours, avec 4 à 24 images par
feu et au moins quatre nombres d’images différents dans le corpus. Tous les contrats, assets,
orthophotos et MNT produits restent dans `/workspace/fireviewer-sdg` et sont vérifiés par SHA-256
avant le lancement. Le wrapper Flow reste dans le volume et référence le preset officiel présent
dans le runtime Isaac épinglé sous `/opt`. Le format exact est décrit dans
[docs/nvidia-real-world-pipeline.md](docs/nvidia-real-world-pipeline.md).

## Frontière avec les reconstructions historiques

Les packs FireViewer de reconstruction rétrospective de juillet 2026 ne sont pas des datasets synthétiques SDG. Les contours `reconstructed` dérivés de sources historiques ne doivent être requalifiés ni comme vérité synthétique, ni comme périmètre observé, ni comme prévision.

## Livrables séparés

Les contrats savent représenter quatre familles de livrables :

| Catégorie | Contrat produit |
| --- | --- |
| `terrestrial_fire_points` | photo terrestre NuRec et exactement trois points `active_fire_point`, `visible_fire_front_point`, `smoke_column_base`, projetés depuis les ancres Flow avec la caméra USD enregistrée |
| `france_cross_view` | photo, position/axe/intrinsics caméra, orthophoto et MNT EPSG:2154 du même site, position du feu inséré vérifiée par l’ancre Flow et le géoréférencement |
| `response_engagement` | une boîte issue de l’AABB d’un acteur USD isolé dans l’image : véhicule SDIS, Canadair, Dash, hélicoptère de Sécurité civile, ou l’un des trois négatifs proches ; les autres acteurs sont invisibles et la géométrie, les proportions et les matériaux restent soumis à la revue |
| `france_incident_days` | dossier fictif A-à-Z avec sources reçues, recherche, faits acceptés/rejetés, contradictions et calque GeoJSON de zone de feu |

Les trois catégories visuelles acceptent une nouvelle capture réelle NuRec ou une nouvelle scène
USD française de référence. Les feux, fumées, acteurs et journées produits restent
synthétiques et sont explicitement marqués comme tels. Aucun cas ne prétend décrire un feu ou un
engagement opérationnel réel.

Le plan Windows `fireviewer-new-synthetic-cases-local-720p-v1` active uniquement
`terrestrial_fire_points`, `france_cross_view` et `france_incident_days`, soit 12 288 cas à la
cible de 4 096 par catégorie et une capacité de 24 576 à 8 192. Véhicules, aéronefs et personnes
restent hors périmètre. Il conserve les mêmes sources,
intrinsics, raycasts USD, seuils de réalisme et contrôles humains, mais écrit les RGB en
1 280 x 720. Son profil et sa révision sont distincts du plan HD : une reprise ne peut donc pas
mélanger silencieusement des images locales 720p avec un lot 1 920 x 1 080.

## Console et revue humaine

Le port 8000 sert `/console`. La coque HTML/CSS est publique, mais statut détaillé, cas, aperçus,
journaux et décisions exigent un jeton Bearer d’au moins 32 caractères. Le jeton reste dans
`sessionStorage`, jamais dans l’URL.

La console ne contient aucun cas, compteur, seed ou journal de démonstration. Elle lit uniquement
les index persistants du pod. Un volume vide affiche uniquement les trois compteurs actifs à zéro.
Les aperçus restent
sur le pod : aucune route de téléchargement du dataset n’est exposée.

Un panneau « Préparation des entrées » expose l'état réel du lock NVIDIA, des trois sites pilotes
et des familles d'assets actives manquantes. Le bouton de production est désactivé côté navigateur **et**
l'API retourne 409 tant que le catalogue complet n'est pas préparé.

Le serveur applique ces verrous :

1. le catalogue de 512 feux et chaque contrat d’événement doivent être complets avant le premier cas ;
2. chaque pilote doit être inspecté, accepté ou rejeté ; une acceptation visuelle exige la liste
   complète des contrôles de réalisme propres à sa catégorie ;
   les contrôles acteur ne s'appliquent que si une campagne future réactive
   `response_engagement` ;
3. le bulk reste bloqué jusqu’à acceptation du pilote de chaque catégorie **et** remplacement du
   périmètre trois-sites par un catalogue géographique étendu marqué `bulk_allowed: true` ;
4. un rejet reste dans l’historique mais ne compte jamais pour la cible ;
5. le bouton de livraison training reste désactivé avant 4 096 acceptations par catégorie ;
6. `POST /v1/training/release` recalcule chaque hash, revalide chaque contrat, contrôle l’unicité
   des seeds et des payloads, au moins 512 feux, les durées 1 à 15 jours, les vues jour/nuit,
   proche/très loin, les occultations bâti/relief et les progressions ;
7. l’audit écrit un JSONL par catégorie active et un manifeste immuables, sans copier ni
   transférer les payloads.

Une acceptation humaine seule ne suffit donc pas : toute corruption ou suppression postérieure
rebloque la livraison.

## Frontière des runtimes et secrets

L’image RunPod contient CUDA, Python 3.12, le bootstrap, le code et les contrats textuels. Au
premier démarrage, elle installe les versions épinglées PyTorch 2.11.0 CUDA 12.8,
Isaac Sim/Replicator 6.0.1.0 et Pillow 12.2.0 sous `/opt/fireviewer-runtime`, sur son disque
conteneur. Rien n’est installé dans `/workspace`.

Le runtime Windows natif est une installation séparée sous `D:\FVS`. Isaac, ses caches et
l’environnement Python restent hors du dépôt ; les entrées et livrables restent sous
`D:\FVS\workspace\fireviewer-sdg`. Aucun runtime WSL n’est requis.

La clé NGC n’est jamais copiée dans le dépôt, l’image, le volume ou les logs. Si une phase NVIDIA
NRE officielle est utilisée, elle doit être fournie au pod comme secret `NGC_API_KEY` ou comme
authentification privée `nvcr.io` avec l’utilisateur `$oauthtoken`.

L’image ne contient ni poids métier, dataset, capture, scène USD externe, texture, secret,
résultat, cas QA ni cache navigateur.

## Construction

```bash
docker build --pull \
  --tag firewarning-datagen:nurec-high-resolution-v1 \
  services/fire-viewer-sdg-worker
```

## Configuration RunPod

### Profil Omniverse Editor 4 × 5

Le profil distinct RunPod cible exclusivement un **RTX PRO 6000 Blackwell
Server Edition 96 Go** avec 150 Go de RAM. Son prévol exige au moins
`90000 MiB` de VRAM et `138000 MiB` de RAM réellement accessibles, en tenant
compte des limites cgroup. Il construit le véritable Omniverse Editor,
matérialise les sources LiDAR et les assets USD/PBR, puis s'arrête avant toute
simulation.

Le portefeuille est fixé à **quatre scènes de base explicitement choisies,
cinq variantes fictives photoréalistes par base, soit vingt scènes**. La
procédure complète et sa frontière de preuve sont documentées dans
[docs/runpod-omniverse-editor-20-simulations.md](docs/runpod-omniverse-editor-20-simulations.md).
Le résultat accepté est verrouillé par le
[contrat normatif photoréaliste](docs/omniverse-photoreal-training-contract.md).

Contrairement au service headless ci-dessous, ce profil utilise 1 500 Go de
NVMe conteneur éphémère, sans volume persistant. Kit, Isaac, Packman, les
sources, les assets, les scènes et les observations restent sur ce disque
jusqu'au transfert vérifié vers une destination choisie sur D:. Aucun script
ne supprime ces données et le pod n'est jamais arrêté sans ordre explicite de
l'opérateur. Le code et les tests locaux ne prouvent pas que le pod, l'Editor,
les vingt scènes, leur qualité RTX ou ce transfert ont été exécutés.

### Service headless historique

| Variable | Défaut | Usage |
| --- | --- | --- |
| `FW_SDG_VOLUME_ROOT` | `/workspace/fireviewer-sdg` | racine persistante des entrées et productions |
| `FW_SDG_RUNTIME_ROOT` | `/opt/fireviewer-runtime/isaacsim-6.0.1.0` | runtime Isaac sur le disque conteneur |
| `FW_SDG_PROVISION_MANIFEST` | manifeste vide de l’image | ressources HTTPS épinglées |
| `FW_SDG_CAMPAIGN` | plan 16 384 cas | contrat de production |
| `FW_SDG_PREPARE_IGN_CATALOG` | `1` | prépare les trois terrains IGN et les 512 contrats si le catalogue est absent |
| `FW_SDG_NVIDIA_ASSET_ROOT` | racine publique Isaac 6.0 épinglée | inventaire automatique en lecture seule des USD officiels |
| `FW_SDG_SIMREADY_ASSET_MANIFEST` | aucun | override local revu pour assets propriétaires ou communautaires licenciés |
| `FW_SDG_AUTH_TOKEN` | aucun | jeton obligatoire en mode service |
| `FW_SDG_STORAGE_MODE` | `network_volume` | `ephemeral` autorise explicitement un disque conteneur non durable |
| `FW_SDG_EPHEMERAL_CAPACITY_GB` | `0` | capacité déclarée, au moins 1000 en mode éphémère |
| `FW_SDG_EPHEMERAL_EXPORT_ACK` | aucun | doit valoir `1` en mode éphémère pour confirmer l'export obligatoire avant arrêt |
| `NGC_API_KEY` | aucun | secret optionnel pour une phase NRE officielle |
| `FW_SDG_ALLOWED_HOSTS` | `huggingface.co` | allowlist de téléchargement |
| `FW_SDG_RUN_MODE` | `service` | `service`, `probe` ou `generate` |
| `FW_SDG_SKIP_GPU_PREFLIGHT` | `false` | uniquement pour le test structurel local |

Le pod expose le port HTTP 8000. Le runtime, ses téléchargements et ses caches restent sous
`/opt`. Les contrats, lockfiles, sources géographiques et livrables restent sous `/workspace`.
La production complète exige 1 000 Go sous `/workspace` : un volume réseau
est le défaut, ou un disque éphémère explicitement déclaré avec export obligatoire avant arrêt.
`GET /healthz` prouve
seulement que le serveur vit. `GET /readyz` authentifié expose les gates GPU et le statut du
contrat réel. Le suivi est disponible via `/v1/console/status`, `/v1/production/status`,
`/v1/logs` et `/v1/cases`.

## Vérifications locales

```powershell
$env:PYTHONPATH='src'
python -m compileall -q src tests
python -m unittest discover -s tests -p 'test_*.py' -v
```

Le test navigateur `tests/console_visual.spec.js` vérifie la console authentifiée avec un cas
réellement indexé et un volume vide. Il est conditionné par `FW_CONSOLE_URL`,
`FW_EMPTY_CONSOLE_URL` et `FW_CONSOLE_TOKEN`.

## Production locale Windows 720p

Prérequis validés : pilote NVIDIA compatible, GPU RTX, runtime Python 3.12 et
`isaacsim[all,extscache]==6.0.1.0` installés dans `D:\FVS\.venv`. Le disque contenant
`D:\FVS\workspace` doit annoncer au moins 1 000 Go et la production garde 100 Go de réserve.

Depuis PowerShell :

```powershell
cd fireviewer-sdg
.\tools\start-local-windows.ps1
```

Le lanceur :

1. crée une fois un jeton aléatoire dans `D:\FVS\config\console-token.txt` sans l’écrire dans
   le dépôt ni les journaux ;
2. force le plan local `fireviewer-new-synthetic-cases-local-720p-v1` ;
3. exécute le checker GPU puis le probe Isaac/Flow/Replicator ;
4. prépare ou reprend les trois sites pilotes et démarre le service seulement après ces gates ;
5. écrit le PID et les chemins des journaux dans `D:\FVS\service.json`.

La console est ensuite disponible sur `http://127.0.0.1:8000/console`. Le jeton doit être saisi
dans le formulaire de connexion ; il n’est jamais placé dans l’URL. Le pilote peut rester bloqué
si les assets SimReady exacts et revus sont incomplets : ce blocage est volontaire et aucun
placeholder ou cas partiel n’est alors produit.

## Portefeuille Omniverse fictif 4 × 5

Le catalogue `livrable_20_zones_france_omniverse` est une source externe en
lecture seule et reste hors Git. Il ne définit pas les vingt scènes finales :
l’opérateur doit fournir exactement quatre de ses IDs dans
`FW_OMNI_BASE_ZONES`, sans sélection automatique, puis cinq compositions sont
produites pour chacune.

Chaque variante conserve les comptes par famille, les identifiants stables,
les références et les échelles des assets. Elle réarrange de façon cohérente la
forêt, les groupes bâtis et les routes. Le terrain utilise un PBR sans arbres,
bâtiments ou routes incrustés : une orthophoto brute ne peut pas servir de
texture visible, afin d’éviter toute imagerie fantôme.

La bibliothèque PBR est partagée, mais les masques, champs de relief et
matériaux restent liés aux payloads de leur tuile. Un graphe shader global
agrégeant les entrées spatiales des 400 tuiles n’est pas un résultat accepté.

Une scène finale possède exactement 400 tuiles, 400 payloads de terrain et
400 payloads d’objets pour chacun des niveaux `HERO`, `MID` et `FAR`. Les tuiles
visibles utilisent `HERO` ou `MID`, le reste garde `FAR`, et le terrain reste
présent sur toute la zone.

Le code prépare actuellement :

- le prévol RTX PRO 6000/96 Go et RAM effective/cgroup ;
- l’environnement LiDAR et le bundle d’assets USD/PBR verrouillé ;
- l’index 4 × 5, le build tuilé de la base pilote et les gates automatiques ;
- la planification et l’authoring natif des vingt variantes via
  `fireviewer_sdg.native_variant_campaign`.

Restent à exécuter sur le pod : les téléchargements réels, les quatre builds de
base, l’authoring des vingt USD et leur contrôle dans le véritable Editor. Une
preuve automatique ne remplace pas cette inspection. Toute tuile vide, rupture
de terrain ou de matériau, forêt anormalement clairsemée, imagerie fantôme,
objet empilé ou flottant, route ou eau incohérente impose le rejet de la scène.

`bash tools/runpod/setup-omniverse-pod.sh prepare` n’ouvre pas l’Editor et ne
lance aucun feu. La revue se lance séparément avec la phase `review`. Une scène
finale complète doit être ouverte et acceptée humainement avant toute
simulation ; l’acceptation est liée par hash au runtime, au catalogue, aux
assets et au root USD courants.

Les LiDAR, datasets, bundles, caches, USD, rendus, résultats et archives restent
sur le volume persistant ou dans le stockage d’artefacts. Les répertoires de
production et les formats lourds sont ignorés par Git ; seuls le code, les
contrats textuels, les tests et les exemples sans secret doivent être suivis.

## Identité et contact

FireViewer est un projet distinct de recherche et développement maintenu par **Unicorn Who Dev**.

> FireViewer n’est ni un service d’alerte, ni une source officielle, ni un outil de conduite des secours. Les sorties et artefacts de ce dépôt exigent leur provenance, leurs gates propres et, lorsqu’ils concernent un incident, une validation humaine.

Contact public, provenance, droits, sécurité et demandes de retrait : [unicornwhodev@gmail.com](mailto:unicornwhodev@gmail.com).
