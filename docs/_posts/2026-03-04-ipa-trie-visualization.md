---
layout: default
title: "02: IPA Trie Visualization"
date: 2026-03-04
categories: [phonology, visualization, linguistics]
---

# IPA Trie Visualization

[Source code](https://github.com/N4D1K-lgtm/research-experiments/tree/main/projects/02-ipa-trie-visualization).

## Setup

The experiment builds a phonological trie from IPA (International Phonetic Alphabet) transcriptions of words across five languages: English (`en_US`), French (`fr_FR`), Spanish (`es_ES`), German (`de`), and Dutch (`nl`). Data comes from [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict), which provides word-to-IPA mappings extracted from Wiktionary.

The pipeline is two-phase: a Python script downloads the TSV data and builds the trie (tokenizing IPA strings, computing a spherical layout, and exporting JSON), then a Three.js web app renders it as an interactive 3D visualization.

### IPA Tokenization

IPA strings are tokenized greedily left-to-right: each phoneme is a base character plus all following combining diacritics and modifiers. Stress marks (ˈˌ), length marks (ː), nasalization, and syllable boundaries are all preserved as part of phoneme identity. Tie bars (◌͡◌) join two bases into a single affricate phoneme.

### Trie Construction

Each word's phoneme sequence is inserted into a prefix trie. At every node traversed, we increment a per-language counter. This means root-level nodes (first phonemes of words) have very high counts, while deep nodes represent rare phoneme sequences. After building, low-frequency nodes are pruned (default: total count < 50) to keep the visualization manageable (~33k nodes from ~1.9M pronunciations).

### Color Blending

The key visual encoding is **circular HSL mean** weighted by per-language counts:

| Language | Hue |
|----------|-----|
| French | 0° (red) |
| Spanish | 12° (red-orange) |
| English | 35° (orange) |
| German | 50° (yellow-orange) |
| Dutch | 62° (yellow) |

Hues are averaged on the unit circle, weighted by word count through each node. Saturation comes from the magnitude of the mean vector — high when one language family dominates, lower when counts are evenly split. This makes language family relationships directly visible: Romance-dominated nodes glow red, Germanic-dominated nodes glow yellow, and nodes shared across families blend to orange.

### Layout

Nodes are arranged in a radial spherical tree: the root sits at the origin, each depth level lies on a concentric spherical shell (radius = depth × 8 units). Root children are distributed via golden angle spiral; deeper nodes subdivide their parent's angular cone proportional to frequency.

## Baseline results

From ~1.9 million pronunciations across 5 languages:

| Language | Pronunciations |
|----------|---------------|
| German | 840,281 |
| Spanish | 595,899 |
| French | 246,465 |
| English | 135,009 |
| Dutch | 121,199 |

After pruning (min count 50): **33,025 nodes**, max depth 16. The bulk of nodes sit at depths 4–10, reflecting typical word lengths in these languages.

### Shared phonological backbone

The highest-frequency nodes near the root — common consonants like /n/, /s/, /t/, /k/, /l/, /m/, /p/ — appear in all five languages and show blended orange colors, forming a shared phonological core. These are the sounds that survived the most mergers across the Indo-European family.

### Language family clustering

Romance languages (French, Spanish) and Germanic languages (German, Dutch) form visible clusters in the trie. Subtrees dominated by one family pull strongly toward their hue. For example:
- French nasal vowels (ɑ̃, ɛ̃, ɔ̃) form deep red subtrees unique to French
- German compound-word phoneme sequences create deep yellow branches extending to depth 16
- English sits between both families, with orange coloring reflecting its mixed Romance-Germanic vocabulary

### Frequency distribution

Node counts follow a steep power law — the top 100 nodes account for the majority of all word paths. The long tail contains rare phoneme sequences, typically from loan words or language-specific morphology.

## Visualization

The web visualization uses Three.js with InstancedMesh for rendering ~33k spheres efficiently. Interactive features:

1. **Language toggles**: Checking/unchecking languages recomputes colors in real-time, showing how the trie shifts between language families
2. **Depth slider**: Reveals the trie layer by layer, from common initial phonemes down to rare sequences
3. **Frequency slider**: Prunes low-frequency branches, highlighting the phonological core
4. **Hover tooltip**: Shows the IPA phoneme, depth, and per-language count breakdown with mini bar charts
5. **IPA labels**: Sprite-based text labels appear for high-frequency nodes near the camera

## Discussion

The visualization reveals that phonological structure is surprisingly shared at shallow depths — the first 3-4 phonemes of words draw heavily from a common inventory across all five languages. Divergence happens at medium depths (5-10), where language-specific phonotactic rules and morphology create branching patterns.

The circular HSL blending is effective for showing language family relationships. When all languages are enabled, the gradient from red through orange to yellow makes the Romance-Germanic continuum visible at a glance. English's position as an orange bridge between the two families is immediately apparent.

### Limitations

- **Data bias**: ipa-dict's coverage varies by language. German's much larger dictionary (840k pronunciations vs 121k for Dutch) means German phoneme sequences are over-represented.
- **Tokenization granularity**: Keeping all diacritics means some nodes represent the same underlying phoneme with different prosodic markings (e.g., stressed vs. unstressed /a/).
- **Layout overlap**: The spherical layout produces some visual overlap at high density, particularly at depths 5-8 where node counts peak.
- **No phonological distance**: Nodes are positioned purely by trie structure, not by articulatory or acoustic similarity. Phonemes that sound similar but appear in different trie positions aren't placed near each other.

## What's next

- **Articulatory-aware layout**: Position nodes using IPA feature vectors (place of articulation, manner, voicing) so phonetically similar sounds cluster spatially
- **Time dimension**: Animate historical sound changes across proto-language reconstruction data
- **More languages**: Add Slavic, East Asian, and Semitic languages to see how the trie and color space extend beyond Indo-European

## References

- open-dict-data/ipa-dict: [github.com/open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) (CC BY-SA)
- International Phonetic Association. *Handbook of the International Phonetic Association*. Cambridge University Press, 1999.
