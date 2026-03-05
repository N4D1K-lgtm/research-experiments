# 02: IPA Trie Visualization

3D visualization of a phonological trie built from IPA (International Phonetic Alphabet) dictionary data across five languages. Each node represents an IPA speech sound; colors blend by language family using circular HSL means, revealing shared phonological structure.

## Languages

- **French** (`fr_FR`) — red (hue 0°)
- **Spanish** (`es_ES`) — red-orange (hue 12°)
- **English** (`en_US`) — orange (hue 35°)
- **German** (`de`) — yellow-orange (hue 50°)
- **Dutch** (`nl`) — yellow (hue 62°)

Color blending uses circular HSL mean: nodes shared across Romance and Germanic languages appear orange, while nodes dominated by one family pull toward red or yellow.

## Quick Start

```bash
# 1. Download IPA dictionary data
python3 scripts/download_data.py

# 2. Build the trie (generates output/trie.json)
python3 scripts/build_trie.py --min-count 50

# 3. Run the web visualization
cd web
npm install
npm run dev
```

## Data Pipeline

- **Source**: [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) (CC BY-SA)
- **Tokenizer**: Greedy left-to-right — base char + combining diacritics/modifiers = one phoneme. All diacritics preserved (stress, length, nasalization, etc.)
- **Trie**: Each node tracks per-language appearance counts (how many words pass through that node)
- **Layout**: Radial spherical tree — root at origin, depth levels on concentric shells, golden spiral distribution
- **Pruning**: `--min-count` flag removes infrequent nodes (recommended: 50+)

## Visualization

- **Three.js** with InstancedMesh for ~33k spheres
- **Color**: Circular HSL mean weighted by language counts — visually reveals language family clustering
- **Interaction**: Hover tooltips with per-language breakdowns, language toggle checkboxes, depth and frequency sliders
- **Labels**: Sprite-based IPA text for high-frequency nodes near camera

## Structure

```
scripts/
  download_data.py    # Fetch TSVs from ipa-dict
  build_trie.py       # Tokenize → trie → layout → JSON
web/
  src/
    main.ts           # Scene + controls + render loop
    types.ts          # TypeScript interfaces
    data/             # Data loading
    rendering/        # Node, Edge, Label renderers
    color/            # Language palette + circular HSL blending
    interaction/      # Tooltip, FilterPanel, Picker
```
