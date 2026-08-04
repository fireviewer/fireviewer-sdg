# Génération des masques et ancrages

## Sorties

- masque flamme ;
- masque fumée ;
- densité ou opacité lorsque disponible ;
- visibilité ;
- occlusion ;
- base de flamme ;
- base de fumée visible ;
- front visible ;
- identifiant d’instance ;
- raisons d’abstention.

## Source de vérité

Les annotations synthétiques proviennent de la scène, des ancres Flow, de la caméra et des buffers de rendu, pas d’une prédiction appliquée après génération.

## Multi-instance

Chaque foyer ou colonne possède un identifiant distinct. Un front peut être une polyligne.

## Abstention

Les cas doivent inclure :

- base masquée ;
- fumée sans origine visible ;
- ambiguïté entre plusieurs ancrages ;
- phénomène hors champ ;
- occultation par relief, bâtiment ou végétation.

## Export

Chaque annotation conserve :

- seed ;
- scène ;
- caméra ;
- révision ;
- contrats ;
- empreintes ;
- liens vers RGB, profondeur et masques.
