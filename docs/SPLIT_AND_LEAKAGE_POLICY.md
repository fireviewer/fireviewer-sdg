# Politique de splits et prévention des fuites

## Groupes

Les splits sont définis par groupes :

- incident synthétique ;
- zone ;
- scène ;
- séquence ;
- capture ;
- source réelle ;
- famille de rendu.

## Interdictions

Un même feu, site, rendu dérivé ou capture proche ne doit pas apparaître dans plusieurs splits.

Les variantes suivantes restent dans le même groupe :

- crop ;
- changement de résolution ;
- augmentation ;
- autre frame proche ;
- autre rendu de la même pose ;
- changement léger de météo ;
- dérivé d’un même média.

## Denylists

Les incidents opérationnels actifs et leurs dérivés sont exclus de l’entraînement et de la validation.

## Lots critiques

Les lots critiques :

- restent hors entraînement ;
- sont versionnés ;
- sont doublement vérifiés lorsque nécessaire ;
- ne sont pas remplacés silencieusement.

## Rapport

Chaque release de dataset publie :

- méthode de groupement ;
- statistiques de provenance ;
- contrôles de duplication ;
- exclusions ;
- empreintes ;
- limites connues.
