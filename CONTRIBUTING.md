# Contribuer à FireViewer SDG

## Règle principale

Les contributions utilisent uniquement des données fictives ou des données publiques dont la
provenance, la licence, la précision et les restrictions sont documentées. Ne soumettez aucune
preuve d'incident réel, position sensible, clé, jeton, secret, capture, scène ou asset dépourvu de
métadonnées vérifiables.

## Périmètre des contributions

Les priorités inter-dépôts sont maintenues dans le dépôt documentaire FireViewer. Dans ce dépôt,
une contribution doit correspondre à une issue SDG actuelle et respecter les responsabilités du
générateur synthétique :

1. contrats de scènes, cas, annotations, masques et ancrages ;
2. provenance des sources, assets, runtimes et livrables ;
3. séparation des splits et prévention des fuites ;
4. gates automatiques, revue humaine et abstentions explicites ;
5. reprise déterministe des campagnes sans publication automatique.

Les datasets, médias, poids, captures NuRec, scènes USD, caches, rendus, résultats et archives
restent hors Git. Les identifiants historiques déjà publiés ne doivent pas être renommés sans
vérifier tous leurs consommateurs.

## Licence des contributions

Tout code soumis à ce dépôt est proposé sous **AGPL-3.0-or-later**. Toute documentation, roadmap,
illustration ou diagramme soumis est proposé sous **CC BY 4.0**. En ouvrant une contribution, vous
confirmez disposer des droits nécessaires pour accorder cette licence et conserver les
attributions requises.

## Avant une pull request

- Décrivez le comportement et les preuves de validation, pas seulement les fichiers modifiés.
- Gardez les changements ciblés et n'introduisez aucune donnée réelle ou dépendance non justifiée.
- Exécutez les contrôles pertinents et indiquez explicitement ceux qui n'ont pas été exécutés.
- Vérifiez que `git status` ne contient aucun `.env`, secret, build, cache, dataset ou archive reçue.
- Exécutez un détecteur de secrets sur le diff et l'historique avant toute publication publique.
- Utilisez une adresse Git `noreply` si votre adresse personnelle ne doit pas apparaître dans
  l'historique public.
- Inscrivez chaque asset, source ou document externe dans [PROVENANCE.md](PROVENANCE.md), avec sa
  licence et sa preuve d'origine.

## Convention de sécurité et de preuve

Une génération réussie ne prouve ni le réalisme, ni la qualité terrain, ni l'absence de fuite, ni
l'acceptation humaine. Les validations automatiques, la revue visuelle et la livraison training
restent des gates distinctes. Aucun placeholder, cas incomplet ou résultat non revu ne doit être
présenté comme un livrable accepté.
