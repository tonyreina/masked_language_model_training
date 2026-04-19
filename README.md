# Masked Language Model Training Animation

An animated visualization of how BERT-style masked language modeling works. Word embeddings start scattered at random and migrate into semantic clusters as training progresses — showing, visually, how the model learns that "milk" and "juice" belong together.

<video src="bert_training.mp4" controls width="100%"></video>

## What it shows

- **Training examples** — 17 epochs cycle through varied sentences, each with one word hidden behind a `[MASK]` token
- **Benchmark sentence** — every 4th epoch replays the same sentence ("I went to the store to buy some `[MASK]`.") so you can track progress against a fixed reference
- **Embedding space** — 36 words across 6 semantic clusters (Drinks, Food, Places, Actions, Animals, Weather) migrate from random positions to their learned cluster positions over training
- **Star marker** — follows the current target word as it moves into its cluster
- **Loss + guess** — the model's current prediction and cross-entropy loss update each frame

The animation is designed for classroom and presentation use: 1920×1080, 24 fps, ~40 seconds.

## "Wheel of Fortune" analogy

Think of MLM training as Wheel of Fortune, but for whole words instead of letters. The model sees the full sentence with one word blanked out and has to guess what fits. Over many examples, it learns that context words like "store" and "buy" point toward consumer goods — and eventually gets "milk" right.

## Usage

```bash
pixi run python animation.py
```

Output is saved as `bert_training.mp4` in the project directory. Rendering prints a `tqdm` progress bar.

## Customization

All tunable parameters are named constants at the top of [animation.py](animation.py):

| Section | Constants | What they control |
|---|---|---|
| `# palette` | `BG`, `ACCENT`, `CLUSTER_COLORS`, … | All colors |
| `# layout / render` | `FIG_W`, `FIG_H`, `DPI`, `CANVAS_W/H` | Output resolution and data-space size |
| `# timing` | `FPS`, `HOLD_FRAMES`, `TRANSITION_FRAMES` | Animation speed |
| `# halo / label fade` | `HALO_FADE_START`, `HALO_MAX_ALPHA`, … | When and how strongly cluster glows appear |
| `# font sizes` | `FS_TITLE`, `FS_DOT_LABEL`, `FS_NARRATION`, … | Every font size in the animation |

### Font size constants

| Constant | Default | Controls |
|---|---|---|
| `FS_TITLE` | 30 | Main header |
| `FS_SUBTITLE` | 16 | Header subtitle |
| `FS_PANEL_LABEL` | 18 | Section labels ("EMBEDDING SPACE", "SEMANTIC CLUSTERS") |
| `FS_CARD_LABEL` | 18 | Card headers ("TRAINING EXAMPLE", "WHAT'S HAPPENING") |
| `FS_EPOCH_PILL` | 13 | Epoch pill |
| `FS_SENTENCE` | 17 | Sentence token text |
| `FS_CHIP` | 15 | `[MASK] → guess` chip |
| `FS_GUESS_LINE` | 12 | Loss / guess label |
| `FS_LEGEND_ITEM` | 18 | Legend cluster names |
| `FS_NARRATION` | 16 | Narration body |
| `FS_DOT_LABEL` | 18 | Word labels in the embedding space |
| `FS_CLUSTER_MAP` | 14 | Cluster names overlaid on the map |

## Requirements

All dependencies are pinned in `pixi.toml` and installed automatically by `pixi`:

| Package | Version |
|---|---|
| Python | ≥ 3.14 |
| matplotlib | ≥ 3.10 |
| numpy | ≥ 2.4 |
| ffmpeg | ≥ 8.0 |
| tqdm | ≥ 4.67 |

Supported platforms: Linux x86-64, macOS (Intel + Apple Silicon), Windows x86-64.
