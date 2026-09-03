# Sync Contracts

The `sicurre` and `sicurre-ml` repositories communicate through two explicit contracts.

## Contract 1: Training dataset

**Direction:** `sicurre` -> `sicurre-ml`

**Canonical store:** Cloudflare R2

**Execution mirror:** Kaggle Dataset

### Expected flow

1. `sicurre` exports a frozen dataset version.
2. The frozen export is written to R2 with immutable version metadata.
3. The MLOps flow packages that exact version into a Kaggle Dataset release for training.
4. Kaggle training consumes the packaged dataset without changing its schema.

### Dataset schema

Training exports must provide CSV files with these columns:
- `text`
- `label`

### Required metadata

Each frozen dataset version should be traceable through:
- dataset version identifier
- export timestamp
- source snapshot or lineage reference
- schema version
- class distribution summary

## Contract 2: Production model

**Direction:** `sicurre-ml` -> `sicurre`

**Store:** Hugging Face Hub

The app repo pins a specific model revision rather than following a floating latest tag.

## Contract 3: Golden evaluation set

**Direction:** `sicurre` -> `sicurre-ml`

**Canonical store and owner:** Sicurre / Cloudflare R2

Sicurre creates, reviews, validates, versions, and publishes the golden set.
Sicurre-ML receives only an immutable evaluation reference and read-only access.
It must not generate, edit, relabel, or republish golden-set records.

The evaluation reference provides:

- dataset ID and immutable version;
- content checksum and schema version;
- provenance and human-review status;
- object reference or evaluation-only mirror; and
- label and language counts.

Questionable records are reported to Sicurre by stable sample ID. Corrections
produce a new immutable version; Sicurre-ML never patches the consumed version.

## Contract 4: Inference mail context

**Direction:** `sicurre` -> `sicurre-ml`

Sicurre derives four bounded booleans from the intercepted message:
`structured_forward`, `outer_sender_authenticated`, `mailing_list_headers`, and
`subscription_claimed`. Raw authentication headers, recipient identities, and
subscription text never enter this context object.

The context may resolve low-phishing-risk `spam` versus `legitimate`
ambiguity. It cannot reduce the phishing probability, bypass URL reputation,
or override a known-malicious URL. A subscription claim without a structured
forward remains untrusted content and never self-whitelists a sender.

## Contract rules

- Do not add direct database or service coupling between repos.
- Do not make Kaggle the canonical data store.
- Do not change schema or model handoff conventions without an ADR in this repo and coordinated updates in `sicurre`.
- Do not use a training/test split as authority for automatic production
  promotion. Follow `docs/model/promotion-policy.md`.
