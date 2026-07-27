# Contrats d’entrée NVIDIA NuRec et Omniverse SimReady

Ce document décrit les données minimales attendues avant toute génération. Le validateur de
référence est `fireviewer_sdg.real_world.load_real_world_contract`; ce texte ne remplace pas le
contrôle exécuté.

## Sens du pipeline

La construction des trois sites pilotes est **real-to-sim** : emprise IGN réelle, orthophoto,
MNT, repère Lambert-93, scène USD métrique et assets SimReady verrouillés. Les feux et engagements
restent fictifs. La phase **sim-to-real** est une évaluation séparée sur des photos françaises
réelles tenues à l'écart ; elle est nécessaire avant de prétendre que les cas synthétiques
améliorent un modèle opérationnel.

Les trois sites sont un périmètre de preuve du setup, pas la couverture géographique finale.

## Découverte et verrouillage des assets

Sans override, `omni.client.list` inventorie uniquement la racine publique versionnée Isaac Sim
6.0. Chaque USD retenu est référencé par un wrapper USDA local et enregistré dans
`simready-assets-hd-v2.json` avec son URI, son identité fournisseur, sa licence et son hash local.
Les caches distants restent sous `/opt/fireviewer-cache`.

La sélection automatique exige six végétations distinctes et un bâtiment rural. Les
correspondances nominales exactes pour SDIS, Canadair, Dash et hélicoptère de Sécurité civile ne
sont exigées que par une campagne qui active `response_engagement`.
`fire_truck` ou `helicopter` génériques ne satisfont pas ces rôles. Les assets communautaires,
commerciaux ou possédés par le projet passent par un manifeste local revu ; aucune provenance ou
licence n'est inférée.

Le décor et ses métriques USD sont validés avant Flow. La V1 Windows exclut véhicules, aéronefs
et personnes : ses contrats portent `scope.response_engagement: false` et une liste d'acteurs
vide. Une campagne future qui active cette catégorie reste fail-closed si les acteurs exacts
manquent.

## Provenance de scène

Le bloc `capture` accepte deux provenances explicites :

- `new_real_world_capture`, avec le chemin NuRec/NRE ci-dessous ;
- `new_synthetic_french_reference`, avec au moins 12 références 4K, une échelle terrain, des
  matériaux et une cohérence orthophoto/MNT validés.

Pour une capture réelle, le bloc doit fournir :

- `source: new_real_world_capture` ;
- `capture_manifest` et `capture_manifest_sha256` ;
- au moins 100 images, dont 95 % enregistrées ;
- `minimum_source_resolution` au moins `[3840, 2160]` ;
- `mean_reprojection_error_px` entre 0 et 1 ;
- `overlap_validated`, `intrinsics_validated`, `extrinsics_validated` et
  `timestamps_validated` à `true` ;
- `coordinate_convention: ncore_rig_and_camera_v4`.

Les conventions NCore sont : repère rig X avant, Y gauche, Z haut ; repère caméra X droite,
Y bas, Z avant. Un repère monde local doit être utilisé pour éviter la perte de précision.

## Reconstruction

Le bloc `reconstruction` accepte `nv-tlabs/3dgrut` ou `nvidia/nre`, au format `particle_field` ou
`nurec_usdz`. Pour une scène synthétique, il exige `nvidia/omniverse_simready` et
`simready_usd`. L’asset est toujours vérifié par SHA-256.

Les métriques doivent provenir d’au moins 10 vues tenues à l’écart :

- `held_out_evaluation: true` ;
- `held_out_view_count >= 10` ;
- `psnr >= 25` ;
- `ssim >= 0.90`.

Pour le chemin COLMAP officiel 3DGRUT, la configuration de production est
`apps/colmap_3dgut_mcmc.yaml` avec `export_usd.enabled=true`. ParticleField est préféré au format
NuRec USDZ propriétaire pour les nouveaux assets Isaac Sim.

## Composition Omniverse

Le stage racine reste un USD/USDA/USDC inscriptible. La reconstruction est référencée sous
`/World/RealWorldScene`; un USDZ NuRec ne doit pas être ouvert directement comme stage racine.
La scène Flow est ajoutée comme payload sous `/World/FireAndSmoke`.

Chaque contrat d’événement exige :

- une scène Flow hachée et une frame simulée validée pour chaque état de progression ;
- exactement les ancres `active_fire_point`, `visible_fire_front_point` et
  `smoke_column_base` ;
- 2 à 8 points de vue ayant passé la calibration NCore, le raycast USD et soit une image de
  référence approuvée, soit une validation en attente dans la console ;
- des distances proche, moyenne, lointaine et très lointaine, des occultations partielles par
  bâti et relief, et des éclairages jour, nuit, aube et crépuscule ;
- des états ordonnés de progression incluant zone de flammes progressante, division de front et
  reprise, avec surface brûlée non décroissante et fronts actifs tracés ;
- si et seulement si `scope.response_engagement` vaut `true`, les quatre acteurs positifs et trois
  négatifs proches, chacun avec asset haché, transform, AABB monde, contexte d’engagement et au
  moins une pose validée ; sinon `composition.actors` doit être vide.

Avant le premier rendu d'un actif NuRec, `nurec_utils.setup_for_rendering(stage)` configure le
renderer NuRec. Cette étape n'est pas importée pour les scènes USD SimReady inspirées du réel,
qui ne contiennent pas de reconstruction NuRec.
Le multi-GPU est désactivé, ACES est activé, l'éclairage extérieur et le film ISO sont calibrés,
Flow est chauffé sur 48 mises à jour puis figé, et Replicator capture en 1 920 x 1 080 avec
16 sous-trames RTX pour le plan livré. Une validation d'histogramme et de composition refuse les
images
surexposées, sous-exposées ou sans dynamique et tente au plus trois expositions bornées avant
un échec fail-closed. Elle refuse aussi la fumée globale, les détails du décor effacés et les
amas émissifs détachés ressemblant à un faux soleil.
Chaque lot visuel est aligné sur les frontières d'événement et ne contient qu'un seul feu. Le
processus utilise le stage courant inscriptible exposé par `SimulationApp.context` et ne rouvre
jamais `UsdContext`. Il attend plusieurs mises à jour consécutives sans chargement ni streaming,
compose USD/Flow timeline arrêtée, attend à nouveau la stabilité complète du stage, termine les
pas de chauffe, puis initialise le graphe Replicator. Il s'arrête avant le feu suivant afin de ne
jamais réutiliser un graphe SDGPipeline ou rouvrir un stage sous les extensions de graphe actives.

## Géospatial France

Le bloc `geospatial` doit déclarer `EPSG:2154`, une origine Lambert-93, l’alignement des axes,
une orthophoto, un MNT et un aperçu MNT. Chaque fichier est obligatoire, confiné au volume et
vérifié par SHA-256. Le profil de paysage doit être rural, montagneux ou agricole français (ou une
combinaison), issu d’une capture française ou d’une référence synthétique explicitement marquée.
La position du feu livré est celle de l’ancre Flow transformée dans ce repère, jamais une position
inférée ou publique.

## Catalogue multi-feux

Le fichier `event-catalog-4096-hd-v2.json` planifie exactement 4 096 cas par catégorie sur au moins 512
feux distincts. Chaque feu dure de 1 à 15 jours et fournit entre 4 et 24 vues ; le nombre varie
réellement dans le corpus. Chaque feu couvre toutes ses poses opérationnelles, jour et nuit,
proche et très loin, les occultations bâti/relief et les trois progressions obligatoires. Les
durées 1 à 15 jours et les douze secteurs d’azimut sont couverts au niveau du catalogue.
La chronologie Flow, la direction et le pas de propagation, le vent, l’intensité et la position
du front sont dérivés indépendamment pour chaque feu ; les 512 signatures de progression doivent
être distinctes avant l’ouverture du pilote.

## Livraison training

Une release ne peut contenir que des cas acceptés. Lors de sa création, le worker relit tous les
artefacts, recalcule les SHA-256 et revalide les contrats. Il exige 4 096 cas par catégorie, des
seeds et payloads principaux uniques, au moins 512 feux et une distribution à un cas près entre
les sept classes d’intervention/négatifs. Un point de vue `pending_console_review` ne devient
livrable qu’après acceptation humaine du cas dans la console.

La release contient uniquement des manifestes locaux ; `transfer_performed` reste `false`.
