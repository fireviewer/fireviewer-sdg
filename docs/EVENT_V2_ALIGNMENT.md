# Alignement avec l’architecture événementielle v2

## Rôle du dépôt

FireViewer SDG produit des fixtures et corpus synthétiques explicitement
identifiés pour compléter les données réelles. Il ne crée pas d’observation
active, d’incident public, de périmètre constaté ni de preuve officielle.

Dans l’architecture événementielle v2, ses sorties servent uniquement à :

- tester les contrats de perception, localisation, incertitude et abstention ;
- compléter un entraînement après validation réel/synthétique ;
- fournir des scènes contrôlées pour le raycast et le recalage ;
- reproduire des cas difficiles sans exposer de contribution privée ;
- évaluer les branches shadow sur des lots séparés des incidents réels.

## Contrat événementiel synthétique

Chaque journée ou séquence synthétique doit conserver un identifiant d’incident
synthétique, une timeline, les poses et intrinsics, la révision de scène, les
seeds, les phénomènes simulés, les masques, les ancrages, la visibilité,
l’occlusion et les motifs d’abstention.

Les objets restent marqués `simulation` dans toutes les sorties. Une simulation
ne peut pas corroborer une observation réelle, fermer une enveloppe d’activité
constatée ou alimenter directement la carte publique.

## Gates avant utilisation

Une famille synthétique n’entre dans un train qu’après :

1. validation des contrats et de la provenance ;
2. contrôle des fuites par incident, zone, scène et séquence ;
3. comparaison avec un lot réel tenu à l’écart ;
4. mesure des gains et régressions par domaine ;
5. revue humaine et décision documentée.

Les entraînements longs restent suspendus tant que le benchmark événementiel
réel ne mesure pas la localisation en mètres, la cohérence temporelle, la
calibration et l’abstention. Le volume brut ou le réalisme visuel ne constitue
pas un gate de promotion.

## Documentation canonique

La doctrine produit et les contrats transverses sont maintenus dans
`fireviewer/Fireviewer_doc`. Les documents de ce dossier décrivent les règles
propres aux datasets synthétiques ; les contrats de campagne existants restent
des documents techniques distincts.
