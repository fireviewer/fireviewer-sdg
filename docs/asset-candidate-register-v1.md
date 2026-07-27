# Registre des assets candidats — V1 locale

Ce registre est un inventaire de recherche, pas un manifeste de production. Aucun asset de cette
liste n'est admis dans un rendu tant que son fichier local, sa licence, son SHA-256, ses dépendances
et ses métriques USD n'ont pas été verrouillés dans `simready-assets-hd-v2.json`.

## Périmètre

La V1 Windows active uniquement :

- `terrestrial_fire_points` ;
- `france_cross_view` ;
- `france_incident_days`.

Les véhicules, aéronefs, personnes et `response_engagement` sont hors périmètre. Leur absence ne
bloque pas la V1 et aucun asset générique ne doit être utilisé pour les simuler.

## Candidats prioritaires

| Famille | Source candidate | Utilisation V1 | Licence connue | État |
| --- | --- | --- | --- | --- |
| végétation USD NVIDIA | `NVIDIA > Assets > Vegetation` et `Assets/Isaac/SimReady`, interrogés depuis Isaac Sim 6 | arbres, buissons et herbes de fond | NVIDIA, à archiver depuis le paquet/runtime exact | découverte Kit et revue visuelle requises |
| pin réaliste | [Poly Haven — Pine Tree 01](https://polyhaven.com/a/pine_tree_01) | silhouettes proches et lointaines, variantes LOD obligatoires | CC0 | candidat fort ; 17 M triangles en source, optimisation obligatoire |
| herbe réaliste | [Poly Haven — Grass Medium 01](https://polyhaven.com/a/grass_medium_01) | sol agricole/rural | CC0 | candidat fort ; LOD et instancing obligatoires |
| sous-bois | [Poly Haven — Fern 02](https://polyhaven.com/a/fern_02) et [Pine Roots](https://polyhaven.com/a/pine_roots) | rupture de répétition au premier plan | CC0 | candidats, densité à limiter |
| rochers | [Poly Haven — Rock Moss Set 02](https://polyhaven.com/a/rock_moss_set_02) | relief, occlusion et diversité montagne | CC0 | candidat fort |
| sol forestier sec | [Poly Haven — Forest Ground 04](https://polyhaven.com/a/forest_ground_04) | sol rural/forestier et base du sol brûlé | CC0 | candidat fort, conversion MaterialX/MDL à valider |
| piste/chemin | [Poly Haven — Rocky Trail](https://polyhaven.com/a/rocky_trail) | chemins DFCI et abords agricoles | CC0 | candidat fort |
| murs ruraux | [Poly Haven — Plastered Stone Wall](https://polyhaven.com/a/plastered_stone_wall) et [Stone Wall 05](https://polyhaven.com/a/stone_wall_05) | matériaux du bâti rural d'occlusion | CC0 | candidats forts ; la géométrie du bâtiment reste à valider |
| matériaux PBR | [NVIDIA vMaterials 2](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) | sol, pierre, bois, plâtre, route | NVIDIA, paquet gratuit annoncé pour les projets | paquet 5,5 Go ; sous-ensemble et dépendances à verrouiller |
| ciel/HDRI | [NVIDIA Environments Skies](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) | jour, nuit, aube, crépuscule | NVIDIA, paquet gratuit annoncé pour les projets | paquet 8,9 Go ; quatre ciels minimum à sélectionner |
| HDRI forestier | [Poly Haven — Pine Picnic](https://polyhaven.com/a/pine_picnic) | lumière diffuse de forêt/pinède | CC0 | candidat jour couvert ; ne couvre pas seul les quatre périodes |
| feu/fumée | [NVIDIA Particle Systems et Extension Samples](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) | présélections de référence et exemples Flow | NVIDIA, paquets gratuits annoncés pour les projets | 159 Mo + 900 Mo ; compatibilité Flow 110/Isaac 6 à tester |

Poly Haven confirme explicitement que ses assets sont CC0 et utilisables pour l'entraînement de
modèles : <https://polyhaven.com/license> et <https://docs.polyhaven.com/en/faq>.

## Blocages restant avant le pilote

1. Le lock NVIDIA n'existe pas encore dans `D:\FVS\workspace\fireviewer-sdg\input`.
2. Aucun fichier candidat ci-dessus n'a encore été téléchargé, converti en USD puis validé dans
   Isaac Sim.
3. Il faut choisir un bâtiment rural crédible pour les occultations. Une coque métrique avec
   matériaux PBR peut suffire au pilote uniquement après comparaison visuelle avec une référence
   française ; un modèle stylisé ou low-poly est refusé.
4. Les presets feu/fumée doivent être rendus dans les quatre éclairages et passer les contrôles
   anti-fumée globale, anti-amas émissif et conservation du détail terrain.
5. Chaque asset admis doit avoir : source stable, licence archivée, SHA-256, mètres/Z-up, AABB,
   textures résolues, budget triangles/VRAM, LOD, instance testée et image de référence acceptée.

## Rejets automatiques

- asset `NoAI`, éditorial ou dont les droits d'entraînement sont ambigus ;
- modèle généré par IA sans provenance et droits vérifiables ;
- style low-poly, jouet, diorama ou proportions non réalistes ;
- texture absente, chemin absolu cassé, unité inconnue ou géométrie sans AABB valide ;
- soleil ou source émissive simulant involontairement une flamme ;
- fumée couvrant la scène sans colonne et base localisées ;
- répétition visible d'un même arbre, rocher ou motif de texture.
