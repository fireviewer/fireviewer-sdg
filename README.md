# FireViewer SDG

**Optional synthetic-data, simulation and real-to-synthetic validation research for FireViewer.**

This repository is a specialised R&D workstream. It explores controlled synthetic cases, simulation-assisted datasets and real/synthetic validation for FireViewer perception and localisation research.

**It is not a runtime dependency of the FireViewer core platform.**

The canonical FireViewer map builder, incident model, temporal fire layers and replay architecture are documented in [`fireviewer/Fireviewer_doc`](https://github.com/fireviewer/Fireviewer_doc).

> Synthetic or simulated data from this repository must never be presented as a real wildfire observation, an official incident record or a propagation forecast.

## Why this repository exists

Real wildfire data is difficult to collect in a balanced way. Some important failure cases are rare, dangerous, poorly labelled or unavailable with the geometric information needed for evaluation.

FireViewer SDG is used to investigate whether carefully controlled synthetic material can help with tasks such as:

- fire/smoke detection evaluation;
- visual-anchor generation;
- cross-view localisation research;
- segmentation and masks;
- camera/geometry validation;
- hard negatives and occlusion cases;
- controlled incident-day fixtures;
- real-to-synthetic / synthetic-to-real comparison.

The objective is **not** to claim that photorealistic rendering solves the real-world domain gap. Synthetic data must be evaluated against held-out real material before it can support a training or benchmark decision.

## Relationship to FireViewer core

```text
FireViewer core
  ├── evidence + provenance
  ├── headless measured map builder
  ├── AI analysis / localisation / abstention
  ├── human review
  ├── temporal fire states
  └── replay / post-event studies

FireViewer SDG (this repo)
  └── optional synthetic-data / simulation research
          ↓
     training or evaluation candidates
          ↓
     independent real-data validation
```

The SDG workstream cannot replace measured map production, historical evidence or reviewed incident state.

## NVIDIA / Omniverse research path

Some experiments in this repository use NVIDIA technologies such as:

- Omniverse / OpenUSD;
- Isaac Sim / Replicator;
- NuRec / NCore-related workflows;
- Flow-based fire/smoke simulation;
- RTX rendering.

These technologies are **experiment-specific dependencies**, not FireViewer-wide dependencies.

Detailed NVIDIA workflows remain documented separately so the repository README does not make them look like the main FireViewer architecture:

- [`docs/nvidia-real-world-pipeline.md`](docs/nvidia-real-world-pipeline.md)
- [`docs/omniverse-photoreal-training-contract.md`](docs/omniverse-photoreal-training-contract.md)
- [`docs/runpod-omniverse-editor-20-simulations.md`](docs/runpod-omniverse-editor-20-simulations.md)

## Synthetic output families

The repository can represent several bounded research families, including:

| Family | Purpose |
| --- | --- |
| `terrestrial_fire_points` | synthetic/controlled terrestrial views with known visible anchors |
| `france_cross_view` | camera + geographic reference pairs for cross-view research |
| `response_engagement` | optional actor/response-object research under separate gates |
| `france_incident_days` | fully synthetic incident-day dossiers for workflow testing |

Each family has its own constraints. A synthetic case is never upgraded to real evidence because it visually resembles a real location.

## Provenance first

Every accepted synthetic case should preserve enough metadata to answer:

- which base scene or geographic reference was used?
- which assets and licences were involved?
- which simulation/render configuration produced the case?
- which camera parameters were used?
- which seed/revision generated the output?
- which labels come directly from scene geometry?
- which quality gates were passed or failed?

Synthetic provenance is part of the dataset, not an optional note.

## Real / synthetic separation

FireViewer keeps strong boundaries between:

```text
real observation
retrospective reconstruction
synthetic case
simulation scenario
training derivative
benchmark result
```

Historical FireViewer reconstruction packs are **not** SDG datasets. A retrospective perimeter derived from real-world sources must not be reclassified as synthetic ground truth.

## Split and leakage policy

Synthetic data is only useful when evaluation remains honest.

Training/evaluation work must preserve separation across relevant dimensions such as:

- incident/scenario identity;
- base scene/location;
- source asset family;
- camera sequence;
- seed/variant lineage;
- real vs synthetic origin.

See [`docs/SPLIT_AND_LEAKAGE_POLICY.md`](docs/SPLIT_AND_LEAKAGE_POLICY.md).

## Validation before training release

A human visual acceptance is not enough to qualify a dataset release.

A release should also pass machine-checkable gates for the applicable family, including:

- contract/schema validity;
- required files present;
- hashes/manifests valid;
- unique/traceable seeds and parents;
- licence/provenance metadata present;
- expected camera/geometric labels consistent;
- no prohibited real/synthetic relabelling;
- split/leakage checks;
- corruption/recovery checks.

See also:

- [`docs/REAL_TO_SYNTHETIC_VALIDATION.md`](docs/REAL_TO_SYNTHETIC_VALIDATION.md)
- [`docs/MASK_AND_ANCHOR_GENERATION.md`](docs/MASK_AND_ANCHOR_GENERATION.md)
- [`docs/DATASET_ROADMAP.md`](docs/DATASET_ROADMAP.md)
- [`docs/EVENT_V2_ALIGNMENT.md`](docs/EVENT_V2_ALIGNMENT.md)

## Repository boundaries

This Git repository contains code, contracts, small fixtures and documentation.

It must not contain:

- production incident media;
- private evidence;
- large generated datasets;
- external proprietary scenes/assets;
- model weights/checkpoints;
- NGC / Hugging Face / cloud credentials;
- runtime caches;
- generated campaign outputs.

Large inputs and outputs live in explicitly configured external storage/runtime roots.

## Current research priority

The current FireViewer-wide priority is **not** to produce the largest possible synthetic corpus.

SDG work should support a specific evaluation gap, for example:

1. define the real-world failure or missing case;
2. create a bounded synthetic experiment;
3. preserve exact provenance;
4. evaluate against held-out real cases;
5. publish failure analysis;
6. only then decide whether the synthetic material belongs in training.

This avoids spending large GPU budgets on synthetic volume without evidence that it improves the target task.

## Licences

The repository code is licensed under AGPL-3.0-or-later and its documentation under CC BY 4.0 where indicated by repository policy.

External NVIDIA software, assets, geographic sources, community assets and other third-party materials retain their own licences and terms. Public availability does not imply redistribution rights.

## Support and collaboration

SDG work can benefit from GPU credits, temporary high-memory GPU access, synthetic-data expertise, graphics/simulation review and independent real-vs-synthetic evaluation.

It is only one part of FireViewer's support needs. See the canonical [Funding Brief](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/FUNDING_BRIEF.md) and [Support & Partnerships](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/SUPPORT_AND_PARTNERSHIPS.md).

## Contact

FireViewer is maintained by **Unicorn Who Dev**.

Research collaboration, infrastructure support, provenance, security and data-removal requests: **unicornwhodev@gmail.com**.
