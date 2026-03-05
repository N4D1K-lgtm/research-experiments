# 02: IPA Trie Visualization

3D visualization of a phonological trie built from IPA (International Phonetic Alphabet) dictionary data across five languages. Each node represents an IPA speech sound; colors blend by language family using circular HSL means, revealing shared phonological structure.

## Languages

- **French** (`fr_FR`) — red (hue 0°)
- **Spanish** (`es_ES`) — red-orange (hue 12°)
- **English** (`en_US`) — orange (hue 35°)
- **German** (`de`) — yellow-orange (hue 50°)
- **Dutch** (`nl`) — yellow (hue 62°)

Color blending uses circular HSL mean: nodes shared across Romance and Germanic languages appear orange, while nodes dominated by one family pull toward red or yellow.

## Prerequisites

- [Rust](https://rustup.rs/) (edition 2024)
- [cargo-make](https://github.com/sagiegurari/cargo-make) (`cargo install cargo-make`)
- Node.js / npm

## Quick Start

```bash
# Full pipeline: download → ingest → build trie → export JSON
cargo make pipeline

# Start the GraphQL server + React frontend
cargo make dev
```

Or step by step:

```bash
cargo make download          # Fetch data (WikiPron, CMU Dict, PHOIBLE)
cargo make ingest            # Ingest into SurrealDB (embedded RocksDB)
cargo make build-trie        # Build trie structures
cargo make export            # Export trie, essay data, and stats as JSON

cargo make server            # Start GraphQL API on :3001
cargo make frontend-dev      # Start React/R3F dev server
```

See `Makefile.toml` for all available tasks and configuration.

## Data Pipeline

**Sources**: [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) (CC BY-SA), WikiPron, CMU Pronouncing Dictionary, PHOIBLE phoneme inventories

**Pipeline** (`ipa-pipeline/`): Rust workspace using SurrealDB (embedded, RocksDB-backed) for storage.

| Crate | Role |
|-------|------|
| `ipa-core` | IPA tokenizer, Unicode normalizer, phoneme inventory |
| `ipa-ingest` | Data download (WikiPron, CMU, PHOIBLE) and ingestion |
| `ipa-db` | SurrealDB schema and queries |
| `ipa-trie` | Trie construction, entropy analysis, motif detection |
| `ipa-export` | JSON export (trie layout, essay data, cross-linguistic stats) |
| `ipa-server` | GraphQL API server |

**Tokenizer**: Greedy left-to-right — base char + combining diacritics/modifiers = one phoneme. Tie bars join affricates. All diacritics preserved (stress, length, nasalization, etc.).

**Layout**: Radial spherical tree — root at origin, depth levels on concentric shells, golden spiral distribution.

## Frontends

Two visualization frontends exist:

- **`frontend/`** — React + React Three Fiber (R3F). Active development. Zustand state management, level-of-detail rendering, search, glow effects, label overlays. Connects to the GraphQL server.
- **`web/`** — Vanilla Three.js + TypeScript. Original prototype. Reads static `trie.json` from `output/`.

## Structure

```
ipa-pipeline/               # Rust workspace
  crates/
    ipa-core/               #   IPA tokenizer + normalizer
    ipa-ingest/             #   Data download + ingestion
    ipa-db/                 #   SurrealDB schema + queries
    ipa-trie/               #   Trie building + analysis
    ipa-export/             #   JSON export
    ipa-server/             #   GraphQL API
  data/                     #   Downloaded source data + SurrealDB store
frontend/                   # React/R3F visualization (active)
web/                        # Vanilla Three.js visualization (prototype)
scripts/                    # Legacy Python scripts
output/                     # Exported JSON (trie, essay, stats)
Makefile.toml               # cargo-make task runner
```
