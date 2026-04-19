"""
Render BERT masked-training animation to MP4.

Two-phase per epoch:
  Phase 1 (game-show) — context words light up one by one; top-5 probability
                        bar chart narrows in real time as each clue appears.
  Phase 2 (embed)     — cut to embedding space; star pulses on the winner.
"""

import math
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
import numpy as np
from tqdm import tqdm

# ── palette ──────────────────────────────────────────────────────────────
BG        = "#0F121C"
PANEL     = "#1A1F2E"
PANEL_LT  = "#22293B"
BORDER    = "#2E364D"
TEXT      = "#F2F3F7"
TEXT_MUTE = "#9AA3B8"
TEXT_DIM  = "#6B748C"
ACCENT    = "#FAC775"
ACCENT_DK = "#854F0B"
MASK_BG   = "#3A2E14"
MASK_FG   = "#FAC775"
BAR_DIM   = "#2A3045"

CLUSTER_COLORS = {
    "drinks":  "#378ADD",
    "food":    "#D85A30",
    "places":  "#1D9E75",
    "actions": "#7F77DD",
    "animals": "#1DC9E7",
    "weather": "#C1D453",
}

# ── layout / render ───────────────────────────────────────────────────────
FIG_W, FIG_H   = 19.2, 10.8
DPI            = 100
CANVAS_W       = 9.0
CANVAS_H       = 4.4

AX_LEFT, AX_BOTTOM, AX_W, AX_H = 0.40, 0.18, 0.55, 0.68

_STAR_X_SCALE = (FIG_H * AX_H / CANVAS_H) / (FIG_W * AX_W / CANVAS_W)

LEFT        = 0.05
RIGHT_LIMIT = 0.95
LINE1_Y     = 0.70
LINE2_Y     = 0.52

# ── timing ────────────────────────────────────────────────────────────────
FPS               = 24
GAMESHOW_FRAMES   = 126  # phase-1 per epoch: word reveal + bar chart  (42 × 3)
EMBED_FRAMES      = 54   # phase-2 per epoch: embedding space focus     (18 × 3)
TRANSITION_FRAMES = 66   # dot migration between epochs                 (22 × 3)
INITIAL_PAUSE     = 90
FINAL_HOLD        = 120

# ── halo / label fade ─────────────────────────────────────────────────────
HALO_FADE_START  = 0.55
HALO_FADE_RANGE  = 0.45
HALO_MAX_ALPHA   = 0.12
LABEL_ALPHA_MIN  = 0.4
LABEL_ALPHA_SPAN = 0.6

# ── font sizes ────────────────────────────────────────────────────────────
FS_TITLE          = 30
FS_SUBTITLE       = 16
FS_PANEL_LABEL    = 18
FS_CARD_LABEL     = 18
FS_EPOCH_PILL     = 13
FS_SENTENCE       = 17
FS_CHIP           = 15
FS_GUESS_LINE     = 12
FS_LEGEND_ITEM    = 18
FS_NARRATION      = 16
FS_DOT_LABEL      = 18
FS_CLUSTER_MAP    = 14
FS_GS_SENTENCE    = 15   # monospace reveal sentence
FS_GS_LABEL       = 13   # "FILL IN THE BLANK" sublabel
FS_BAR_WORD       = 13   # candidate word labels
FS_BAR_VAL        = 11   # probability percentages

# ── dataclasses ───────────────────────────────────────────────────────────
@dataclass
class Cluster:
    id: str
    label: str
    cx: float
    cy: float
    words: list[str]

@dataclass
class Epoch:
    tokens: list[str]
    mask_idx: int
    truth: str
    epoch: int
    is_bench: bool
    guess: str = ""
    loss: float = 0.0

# ── static data ───────────────────────────────────────────────────────────
CLUSTERS: list[Cluster] = [
    Cluster("drinks",  "Drinks",  1.8, 1.2, ["milk","coffee","tea","juice","water","wine"]),
    Cluster("food",    "Food",    1.8, 3.2, ["bread","apple","pasta","cheese","rice","cake"]),
    Cluster("places",  "Places",  4.5, 1.2, ["store","market","park","office","school","cafe"]),
    Cluster("actions", "Actions", 4.5, 3.2, ["buy","read","drive","eat","walk","write"]),
    Cluster("animals", "Animals", 7.2, 1.2, ["dog","cat","bird","horse","fish","mouse"]),
    Cluster("weather", "Weather", 7.2, 3.2, ["rain","snow","sun","wind","cloud","storm"]),
]

_CLUSTER_BY_WORD: dict[str, Cluster] = {w: c for c in CLUSTERS for w in c.words}

_BENCHMARK_TOKENS   = ["I","went","to","the","store","to","buy","some","[MASK]","."]
_BENCHMARK_MASK_IDX = 8
_BENCHMARK_TRUTH    = "milk"

_OTHER_SENTENCES: list[tuple[list[str], int, str]] = [
    (["She","drank","a","glass","of","[MASK]","this","morning","."], 5, "juice"),
    (["The","[MASK]","chased","the","cat","across","the","yard","."], 1, "dog"),
    (["Heavy","[MASK]","fell","throughout","the","afternoon","."],   1, "rain"),
    (["I","love","to","[MASK]","books","on","rainy","days","."],     3, "read"),
    (["She","baked","fresh","[MASK]","for","breakfast","."],         3, "bread"),
    (["They","met","at","a","quiet","[MASK]","downtown","."],        5, "cafe"),
    (["The","[MASK]","sang","outside","my","window","."],            1, "bird"),
]

_BENCH_PROGRESSION: list[tuple[str, float]] = [
    ("cactus", 9.80),
    ("chair",  5.40),
    ("bread",  2.80),
    ("water",  1.20),
    ("milk",   0.25),
]

# ── math helpers (defined early; needed for bar-snapshot generation) ───────
def ease_in_out(t: float) -> float:
    return 3*t*t - 2*t*t*t

def interp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def stitch(tokens: list[str]) -> str:
    out = ""
    for t in tokens:
        if t in (".", ","):
            out = out.rstrip() + t
        else:
            out += t + " "
    return out

def star_vertices(cx: float, cy: float, r_outer: float, r_inner: float, n: int = 5):
    verts = []
    for i in range(2 * n):
        angle = -math.pi / 2 + i * math.pi / n
        r = r_outer if i % 2 == 0 else r_inner
        verts.append((cx + r * _STAR_X_SCALE * math.cos(angle),
                      cy + r * math.sin(angle)))
    return verts

# ── epoch construction ────────────────────────────────────────────────────
def _other_guess_loss(truth: str, epoch_num: int) -> tuple[str, float]:
    rng = random.Random(epoch_num)
    if epoch_num < 20:
        guess = rng.choice(["table","yellow","green","bright","maybe","old","house"])
        loss  = round(8.8 - epoch_num * 0.04, 2)
    elif epoch_num < 60:
        pool  = [w for w in _CLUSTER_BY_WORD[truth].words if w != truth]
        guess = rng.choice(pool)
        loss  = round(4.2 - (epoch_num - 20) * 0.05, 2)
    else:
        guess = truth
        loss  = round(1.3 - (epoch_num - 60) * 0.015, 2)
    return guess, loss

def _build_epochs() -> list[Epoch]:
    epochs: list[Epoch] = []
    bench_idx = other_idx = 0
    for i in range(17):
        epoch_num = i * 10
        if i % 4 == 0:
            g, l = _BENCH_PROGRESSION[min(bench_idx, len(_BENCH_PROGRESSION) - 1)]
            epochs.append(Epoch(
                tokens=_BENCHMARK_TOKENS, mask_idx=_BENCHMARK_MASK_IDX,
                truth=_BENCHMARK_TRUTH, epoch=epoch_num, is_bench=True,
                guess=g, loss=l,
            ))
            bench_idx += 1
        else:
            tokens, mask_idx, truth = _OTHER_SENTENCES[other_idx % len(_OTHER_SENTENCES)]
            guess, loss = _other_guess_loss(truth, epoch_num)
            epochs.append(Epoch(
                tokens=tokens, mask_idx=mask_idx, truth=truth,
                epoch=epoch_num, is_bench=False, guess=guess, loss=loss,
            ))
            other_idx += 1
    return epochs

EPOCHS = _build_epochs()

# ── bar-chart snapshot data ───────────────────────────────────────────────
N_BAR_STEPS = 5   # probability snapshots across the game-show reveal

def _build_bar_snapshots(ep: Epoch) -> list[list[tuple[str, float]]]:
    """N_BAR_STEPS snapshots, each a fixed-order list of (word, prob) pairs.
    Slot 0 is always the model's guess; it ends up dominant."""
    guess       = ep.guess
    guess_clust = _CLUSTER_BY_WORD.get(guess)
    rng         = random.Random(ep.epoch * 31 + 7)

    if guess_clust is not None:
        same        = [w for w in guess_clust.words if w != guess]
        competitors = rng.sample(same, min(2, len(same)))
    else:
        competitors = []

    outsider_pool = [
        w for c in CLUSTERS
        if guess_clust is None or c.id != guess_clust.id
        for w in c.words
    ]
    n_out     = max(0, 4 - len(competitors))
    outsiders = rng.sample(outsider_pool, min(n_out, len(outsider_pool)))

    words = ([guess] + competitors + outsiders)[:5]
    while len(words) < 5:
        words.append(words[-1])

    snapshots = []
    for step in range(N_BAR_STEPS):
        t       = step / (N_BAR_STEPS - 1)
        eased   = ease_in_out(t)
        guess_p = 0.14 + 0.66 * eased
        rem     = 1.0 - guess_p

        raw = {guess: guess_p}
        others = [w for w in words if w != guess]
        for i, w in enumerate(others):
            raw[w] = rem * max(0.04, 1.0 - i * 0.28)

        total = sum(raw.values())
        snapshots.append([(w, raw[w] / total) for w in words])

    return snapshots

EPOCH_BARS: list[list[list[tuple[str, float]]]] = [
    _build_bar_snapshots(ep) for ep in EPOCHS
]

# ── per-word start/end positions ──────────────────────────────────────────
rng_master = random.Random(42)
all_points: list[dict] = []
for c in CLUSTERS:
    for w in c.words:
        wr     = random.Random(hash((c.id, w)) & 0xFFFFFFFF)
        angle  = wr.random() * 2 * math.pi
        radius = 0.35 + wr.random() * 0.55
        end    = (c.cx + radius * math.cos(angle), c.cy + radius * math.sin(angle))
        start  = (0.4 + rng_master.random() * (CANVAS_W - 0.8),
                  0.3 + rng_master.random() * (CANVAS_H - 0.6))
        all_points.append({"word": w, "cluster": c.id, "color": CLUSTER_COLORS[c.id],
                            "start": start, "end": end})

# ── figure setup ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)

# Embedding-space axes (right panel, always visible)
ax_space = fig.add_axes([AX_LEFT, AX_BOTTOM, AX_W, AX_H])
ax_space.set_facecolor(PANEL)
ax_space.set_xlim(0, CANVAS_W)
ax_space.set_ylim(CANVAS_H, 0)
ax_space.set_xticks([]); ax_space.set_yticks([])
for spine in ax_space.spines.values():
    spine.set_color(BORDER); spine.set_linewidth(0.5)

fig.text(0.41, 0.875, "EMBEDDING SPACE (2D projection)",
         color=TEXT_DIM, fontsize=FS_PANEL_LABEL, weight="bold", family="DejaVu Sans")
fig.text(0.04, 0.93,  "BERT learns to fill in the blank",
         color=TEXT, fontsize=FS_TITLE, weight="bold", family="DejaVu Sans")
fig.text(0.04, 0.895, "Masked language modeling — one training example per click",
         color=TEXT_MUTE, fontsize=FS_SUBTITLE, family="DejaVu Sans")

# Epoch pill
pill_ax = fig.add_axes([0.86, 0.905, 0.10, 0.04])
pill_ax.set_xlim(0, 1); pill_ax.set_ylim(0, 1); pill_ax.axis("off")
pill_patch = FancyBboxPatch(
    (0.02, 0.1), 0.96, 0.8,
    boxstyle="round,pad=0.02,rounding_size=0.4",
    linewidth=0.5, edgecolor=BORDER, facecolor=PANEL_LT,
)
pill_ax.add_patch(pill_patch)
pill_text = pill_ax.text(0.5, 0.5, "Epoch 0", ha="center", va="center",
                         color=TEXT_MUTE, fontsize=FS_EPOCH_PILL, family="DejaVu Sans")

# ── sentence card ─────────────────────────────────────────────────────────
sent_ax = fig.add_axes([0.04, 0.58, 0.33, 0.30])
sent_ax.set_xlim(0, 1); sent_ax.set_ylim(0, 1); sent_ax.axis("off")
sent_ax.add_patch(FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0,
    boxstyle="round,pad=0.0,rounding_size=0.02",
    linewidth=0.5, edgecolor=BORDER, facecolor=PANEL,
))
sent_ax.add_patch(Rectangle((0.0, 0.0), 0.018, 1.0, facecolor=ACCENT, edgecolor="none"))

# Embed-phase title (shown during embed / transition)
sent_title = sent_ax.text(0.05, 0.88, "TRAINING EXAMPLE", color=TEXT_DIM,
                          fontsize=FS_CARD_LABEL, weight="bold", family="DejaVu Sans",
                          transform=sent_ax.transAxes)

# Embed-phase sentence artists
sentence_text = sent_ax.text(
    LEFT, LINE1_Y, "", color=TEXT, fontsize=FS_SENTENCE, family="DejaVu Sans",
    transform=sent_ax.transAxes, va="center", ha="left",
)
chip_bg = FancyBboxPatch(
    (0, 0), 0.01, 0.01,
    boxstyle="round,pad=0.006,rounding_size=0.012",
    linewidth=0.5, edgecolor=MASK_FG, facecolor=MASK_BG,
    transform=sent_ax.transAxes, zorder=1,
)
sent_ax.add_patch(chip_bg)
chip_text = sent_ax.text(0, LINE1_Y, "", color=MASK_FG, fontsize=FS_CHIP, weight="bold",
                         family="DejaVu Sans Mono", transform=sent_ax.transAxes,
                         va="center", ha="left", zorder=2)
post_text  = sent_ax.text(0, LINE1_Y, "", color=TEXT, fontsize=FS_SENTENCE,
                          family="DejaVu Sans", transform=sent_ax.transAxes,
                          va="center", ha="left")
post_line2 = sent_ax.text(LEFT, LINE2_Y, "", color=TEXT, fontsize=FS_SENTENCE,
                          family="DejaVu Sans", transform=sent_ax.transAxes,
                          va="center", ha="left")
guess_text = sent_ax.text(0.05, 0.12, "", color=TEXT_MUTE, fontsize=FS_GUESS_LINE,
                          family="DejaVu Sans", transform=sent_ax.transAxes)

# Game-show phase sentence artists
gs_sublabel = sent_ax.text(0.05, 0.88, "FILL IN THE BLANK", color=TEXT_DIM,
                            fontsize=FS_GS_LABEL, weight="bold", family="DejaVu Sans",
                            transform=sent_ax.transAxes, visible=False)
gs_sentence = sent_ax.text(0.05, 0.55, "", color=TEXT, fontsize=FS_GS_SENTENCE,
                            family="DejaVu Sans Mono", transform=sent_ax.transAxes,
                            va="center", visible=False)
# [?] mask indicator shown separately so it can be accent-coloured
gs_mask_word = sent_ax.text(0.05, 0.28, "predict →  [  ?  ]", color=ACCENT,
                             fontsize=FS_GS_LABEL, weight="bold", family="DejaVu Sans Mono",
                             transform=sent_ax.transAxes, va="center", visible=False)

# ── legend ────────────────────────────────────────────────────────────────
leg_ax = fig.add_axes([0.04, 0.44, 0.33, 0.11])
leg_ax.set_xlim(0, 1); leg_ax.set_ylim(0, 1); leg_ax.axis("off")
leg_ax.text(0, 0.92, "SEMANTIC CLUSTERS", color=TEXT_DIM, fontsize=FS_PANEL_LABEL,
            weight="bold", family="DejaVu Sans")
for i, c in enumerate(CLUSTERS):
    col = i % 3; row = i // 3
    lx = 0.02 + col * 0.33
    ly = 0.55 - row * 0.30
    leg_ax.scatter([lx], [ly], s=80, c=CLUSTER_COLORS[c.id], zorder=5, edgecolors="none")
    leg_ax.text(lx + 0.03, ly, c.label, color=TEXT, fontsize=FS_LEGEND_ITEM,
                family="DejaVu Sans", va="center")

# ── narration panel (embed / transition phases only) ──────────────────────
narr_ax = fig.add_axes([0.04, 0.20, 0.33, 0.21])
narr_ax.set_xlim(0, 1); narr_ax.set_ylim(0, 1); narr_ax.axis("off")
narr_ax.add_patch(FancyBboxPatch(
    (0, 0), 1.0, 1.0,
    boxstyle="round,pad=0.0,rounding_size=0.02",
    linewidth=0.5, edgecolor=BORDER, facecolor=PANEL_LT,
))
narr_ax.text(0.05, 0.82, "WHAT'S HAPPENING", color=TEXT_DIM, fontsize=FS_CARD_LABEL,
             weight="bold", family="DejaVu Sans")
narr_text = narr_ax.text(0.05, 0.60, "", color=TEXT, fontsize=FS_NARRATION,
                         family="DejaVu Sans", transform=narr_ax.transAxes,
                         va="top", wrap=True)

# ── probability bar chart (game-show phase only) ──────────────────────────
# Positioned to replace the narration panel during game-show
ax_bar = fig.add_axes([0.04, 0.17, 0.33, 0.24])
ax_bar.set_facecolor(PANEL_LT)
ax_bar.set_xlim(0, 1)
ax_bar.set_ylim(-0.5, 4.5)
ax_bar.set_xticks([]); ax_bar.set_yticks([])
for spine in ax_bar.spines.values():
    spine.set_color(BORDER); spine.set_linewidth(0.5)
ax_bar.text(0.04, 4.15, "TOP CANDIDATES", color=TEXT_DIM,
            fontsize=FS_CARD_LABEL, weight="bold", family="DejaVu Sans")

# 5 bar slots; slot 0 = top (guess word)
_BAR_SLOTS_Y   = [3.35, 2.55, 1.75, 0.95, 0.15]
_BAR_HEIGHT    = 0.52
_BAR_LEFT      = 0.24   # x data-coord where bars start
_BAR_MAX_W     = 0.72   # maximum bar width in data-coords

_bar_rects : list[Rectangle] = []
_bar_words : list            = []
_bar_vals  : list            = []

for slot, y in enumerate(_BAR_SLOTS_Y):
    rect = Rectangle((_BAR_LEFT, y - _BAR_HEIGHT / 2), 0.0, _BAR_HEIGHT,
                     facecolor=BAR_DIM, edgecolor="none", zorder=2)
    ax_bar.add_patch(rect)
    _bar_rects.append(rect)
    _bar_words.append(ax_bar.text(
        _BAR_LEFT - 0.03, y, "", color=TEXT_MUTE, fontsize=FS_BAR_WORD,
        family="DejaVu Sans", ha="right", va="center"))
    _bar_vals.append(ax_bar.text(
        _BAR_LEFT + _BAR_MAX_W + 0.02, y, "", color=TEXT_MUTE, fontsize=FS_BAR_VAL,
        family="DejaVu Sans", ha="left", va="center"))

ax_bar.set_visible(False)   # hidden until game-show phase

# ── embedding-space artists ───────────────────────────────────────────────
cluster_halos: list[tuple] = []
for c in CLUSTERS:
    halo = mpatches.Ellipse(
        (c.cx, c.cy), width=2.0, height=1.6,
        facecolor=CLUSTER_COLORS[c.id], edgecolor="none", alpha=0.0,
    )
    ax_space.add_patch(halo)
    cluster_halos.append((halo, c))

dot_scatter = ax_space.scatter(
    [p["start"][0] for p in all_points],
    [p["start"][1] for p in all_points],
    s=60, c=[p["color"] for p in all_points],
    edgecolors="none", zorder=3,
)
dot_labels: list = []
for p in all_points:
    dot_labels.append(ax_space.text(
        p["start"][0] + 0.15, p["start"][1] - 0.02,
        p["word"], color=TEXT_MUTE, fontsize=FS_DOT_LABEL, family="DejaVu Sans", zorder=4,
    ))

star = Polygon(star_vertices(0, 0, 0.26, 0.11), closed=True,
               facecolor=ACCENT, edgecolor=ACCENT_DK, linewidth=1.5, zorder=6)
ax_space.add_patch(star)

cluster_labels_on_map: list = []
for c in CLUSTERS:
    cluster_labels_on_map.append(ax_space.text(
        c.cx, c.cy - 0.95, c.label.upper(),
        color=CLUSTER_COLORS[c.id], fontsize=FS_CLUSTER_MAP, weight="bold",
        family="DejaVu Sans", ha="center", alpha=0.0, zorder=2,
    ))

# ── timeline ──────────────────────────────────────────────────────────────
TOTAL_FRAMES = (
    INITIAL_PAUSE
    + FINAL_HOLD
    + len(EPOCHS) * (GAMESHOW_FRAMES + EMBED_FRAMES)
    + (len(EPOCHS) - 1) * TRANSITION_FRAMES
)

def frame_to_state(frame: int) -> tuple[int, float, float, str]:
    """Return (epoch_idx, phase_frac, global_t, phase).

    phase_frac  — 0..1 within the current phase
    phase       — 'gameshow' | 'embed' | 'transition'
    global_t    — 0..1 across the whole animation (drives dot positions)
    """
    f = frame - INITIAL_PAUSE
    n = len(EPOCHS)
    if f < 0:
        return 0, 0.0, 0.0, "gameshow"

    for i in range(n):
        if i > 0:
            if f < TRANSITION_FRAMES:
                alpha = f / TRANSITION_FRAMES
                g_t   = ((i - 1) + alpha) / (n - 1)
                return i, alpha, g_t, "transition"
            f -= TRANSITION_FRAMES

        if f < GAMESHOW_FRAMES:
            g_t = i / (n - 1) if n > 1 else 1.0
            return i, f / GAMESHOW_FRAMES, g_t, "gameshow"
        f -= GAMESHOW_FRAMES

        if f < EMBED_FRAMES:
            g_t = i / (n - 1) if n > 1 else 1.0
            return i, f / EMBED_FRAMES, g_t, "embed"
        f -= EMBED_FRAMES

    return n - 1, 1.0, 1.0, "embed"

# ── draw helpers ──────────────────────────────────────────────────────────
def _finalize_guess_label(ep: Epoch) -> None:
    prefix = "★ Benchmark — loss: " if ep.is_bench else "Loss: "
    guess_text.set_text(f"{prefix}{ep.loss:.2f}")
    guess_text.set_color(ACCENT if ep.is_bench else TEXT_MUTE)

def _place_chip(renderer, inv, chip_str: str, chip_left: float) -> float:
    chip_text.set_text(chip_str)
    chip_text.set_position((chip_left + 0.012, LINE1_Y))
    bbox = chip_text.get_window_extent(renderer=renderer)
    return inv.transform((bbox.x1, bbox.y1))[0] + 0.015

def draw_sentence(ep: Epoch, guess_word: str) -> None:
    """Redraw sentence with mask chip; wraps to two lines when needed."""
    pre_tokens  = ep.tokens[:ep.mask_idx]
    post_tokens = ep.tokens[ep.mask_idx + 1:]
    chip_str    = f"[MASK] → {guess_word}"

    renderer = fig.canvas.get_renderer()
    inv      = sent_ax.transAxes.inverted()

    sentence_text.set_fontsize(FS_SENTENCE)
    chip_text.set_fontsize(FS_CHIP)
    post_text.set_fontsize(FS_SENTENCE)
    post_line2.set_fontsize(FS_SENTENCE)

    pre_str = stitch(pre_tokens)
    sentence_text.set_text(pre_str)
    sentence_text.set_position((LEFT, LINE1_Y))
    pre_bbox  = sentence_text.get_window_extent(renderer=renderer)
    chip_left = inv.transform((pre_bbox.x1, pre_bbox.y0))[0] + 0.005

    chip_right = _place_chip(renderer, inv, chip_str, chip_left)
    chip_w     = chip_right - chip_left
    chip_bg.set_bounds(chip_left, LINE1_Y - 0.07, chip_w, 0.14)
    post_left  = chip_right + 0.005

    if post_tokens == ["."]:
        post_text.set_text(".")
        post_text.set_position((post_left, LINE1_Y))
        post_line2.set_text("")
        return _finalize_guess_label(ep)

    post_str_full = stitch(post_tokens)
    post_text.set_text(post_str_full)
    post_text.set_position((post_left, LINE1_Y))
    post_right = inv.transform(
        post_text.get_window_extent(renderer=renderer).get_points()[1]
    )[0]

    if post_right <= RIGHT_LIMIT:
        post_line2.set_text("")
        return _finalize_guess_label(ep)

    for split in range(len(post_tokens), -1, -1):
        line2_tokens = post_tokens[split:]
        if line2_tokens == ["."]:
            continue
        line1_post = stitch(post_tokens[:split])
        post_text.set_text(line1_post)
        post_text.set_position((post_left, LINE1_Y))
        if line1_post.strip():
            right_edge = inv.transform(
                post_text.get_window_extent(renderer=renderer).get_points()[1]
            )[0]
        else:
            right_edge = post_left
        if right_edge <= RIGHT_LIMIT:
            post_line2.set_text(stitch(line2_tokens).rstrip())
            post_line2.set_position((LEFT, LINE2_Y))
            return _finalize_guess_label(ep)

    post_text.set_text("")
    post_line2.set_text(post_str_full.rstrip())
    post_line2.set_position((LEFT, LINE2_Y))
    return _finalize_guess_label(ep)

# ── game-show helpers ─────────────────────────────────────────────────────
def _gs_reveal_string(ep: Epoch, reveal_count: int) -> str:
    """Build the WoF-style sentence: revealed tokens shown, others as underscores."""
    ctx_used = 0
    parts = []
    for i, tok in enumerate(ep.tokens):
        if i == ep.mask_idx:
            parts.append("[?]")
        else:
            parts.append(tok if ctx_used < reveal_count else "_" * len(tok))
            ctx_used += 1
    # Re-join respecting punctuation
    out = ""
    for p in parts:
        if p in (".", ","):
            out = out.rstrip() + p + " "
        else:
            out += p + " "
    return out.strip()

def draw_gameshow_sentence(ep: Epoch, phase_frac: float) -> None:
    n_ctx        = sum(1 for i in range(len(ep.tokens)) if i != ep.mask_idx)
    reveal_count = min(int(phase_frac * (n_ctx + 1)), n_ctx)
    gs_sentence.set_text(_gs_reveal_string(ep, reveal_count))

def draw_bar_chart(epoch_idx: int, ep: Epoch, phase_frac: float) -> None:
    snapshots = EPOCH_BARS[epoch_idx]
    n         = len(snapshots)

    # Interpolate between adjacent snapshots
    pos    = phase_frac * (n - 1)
    lo     = max(0, min(int(pos), n - 2))
    hi     = lo + 1
    t      = pos - lo

    snap_lo = snapshots[lo]
    snap_hi = snapshots[hi]

    guess       = ep.guess
    guess_clust = _CLUSTER_BY_WORD.get(guess)
    win_color   = CLUSTER_COLORS[guess_clust.id] if guess_clust else ACCENT

    for slot in range(5):
        word, prob_lo = snap_lo[slot]
        _,    prob_hi = snap_hi[slot]
        prob    = interp(prob_lo, prob_hi, t)
        is_win  = (word == guess)

        _bar_rects[slot].set_width(prob * _BAR_MAX_W)
        _bar_rects[slot].set_facecolor(win_color if is_win else BAR_DIM)
        color = TEXT if is_win else TEXT_MUTE
        _bar_words[slot].set_text(word)
        _bar_words[slot].set_color(color)
        _bar_vals[slot].set_text(f"{prob:.0%}")
        _bar_vals[slot].set_color(color)

# ── panel-visibility helpers ──────────────────────────────────────────────
def _show_embed_panel(v: bool) -> None:
    sent_title.set_visible(v)
    sentence_text.set_visible(v)
    chip_bg.set_visible(v)
    chip_text.set_visible(v)
    post_text.set_visible(v)
    post_line2.set_visible(v)
    guess_text.set_visible(v)

def _show_gameshow_panel(v: bool) -> None:
    gs_sublabel.set_visible(v)
    gs_sentence.set_visible(v)
    gs_mask_word.set_visible(v)

# ── embedding-space update helpers ────────────────────────────────────────
def update_dots_for_global_t(g_t: float) -> None:
    eased   = ease_in_out(g_t)
    offsets = []
    for i, p in enumerate(all_points):
        x = interp(p["start"][0], p["end"][0], eased)
        y = interp(p["start"][1], p["end"][1], eased)
        offsets.append([x, y])
        dot_labels[i].set_position((x + 0.15, y - 0.02))
        dot_labels[i].set_alpha(LABEL_ALPHA_MIN + LABEL_ALPHA_SPAN * eased)
    dot_scatter.set_offsets(np.array(offsets))

    halo_strength = max(0.0, (eased - HALO_FADE_START) / HALO_FADE_RANGE)
    for halo, _ in cluster_halos:
        halo.set_alpha(HALO_MAX_ALPHA * halo_strength)
    for label in cluster_labels_on_map:
        label.set_alpha(0.75 * halo_strength)

def update_star_for_epoch(ep: Epoch, g_t: float, scale: float = 1.0) -> None:
    target = next(p for p in all_points if p["word"] == ep.truth)
    eased  = ease_in_out(g_t)
    x = interp(target["start"][0], target["end"][0], eased)
    y = interp(target["start"][1], target["end"][1], eased)
    star.set_xy(star_vertices(x, y, 0.26 * scale, 0.11 * scale))

def narration_for(ep: Epoch, epoch_idx: int) -> str:
    if ep.is_bench:
        if epoch_idx == 0:
            return ("We'll use this sentence as a recurring benchmark.\n"
                    "The word 'milk' is hidden. BERT has to guess it\n"
                    "from the surrounding context.")
        if ep.guess == ep.truth:
            return ("BERT has converged on the benchmark sentence.\n"
                    "The guess is correct and loss is near zero.\n"
                    "Geometry has done the work.")
        return ("Benchmark checkpoint — same sentence as before,\n"
                "but BERT's guess has shifted as the embedding\n"
                "space has reorganized.")
    return ("A new training example. One word is hidden;\n"
            "BERT must predict it from context.\n"
            "Gradients nudge the geometry toward truth.")

# ── main animation update ─────────────────────────────────────────────────
def update(frame: int):
    epoch_idx, phase_frac, g_t, phase = frame_to_state(frame)
    ep = EPOCHS[epoch_idx]

    # Always update dots and pill
    update_dots_for_global_t(g_t)

    pill_text.set_text(f"Epoch {ep.epoch}")
    pill_patch.set_facecolor(MASK_BG if ep.is_bench else PANEL_LT)
    pill_patch.set_edgecolor(ACCENT  if ep.is_bench else BORDER)
    pill_text.set_color(ACCENT       if ep.is_bench else TEXT_MUTE)

    if phase == "gameshow":
        # ── Phase 1: context clues light up, bar chart narrows ────────────
        update_star_for_epoch(ep, g_t)  # star at normal size

        _show_embed_panel(False)
        _show_gameshow_panel(True)
        narr_ax.set_visible(False)
        ax_bar.set_visible(True)

        draw_gameshow_sentence(ep, phase_frac)
        draw_bar_chart(epoch_idx, ep, phase_frac)

    elif phase == "embed":
        # ── Phase 2: cut to embedding space; star pulses on winner ────────
        pulse = 1.0 + 0.55 * math.sin(phase_frac * math.pi)
        update_star_for_epoch(ep, g_t, scale=pulse)

        _show_embed_panel(True)
        _show_gameshow_panel(False)
        narr_ax.set_visible(True)
        ax_bar.set_visible(False)

        draw_sentence(ep, ep.guess)
        narr_text.set_text(narration_for(ep, epoch_idx))

    else:  # transition — dots migrate, keep embed layout
        update_star_for_epoch(ep, g_t)

        _show_embed_panel(True)
        _show_gameshow_panel(False)
        narr_ax.set_visible(True)
        ax_bar.set_visible(False)

        draw_sentence(ep, ep.guess)
        narr_text.set_text(narration_for(ep, epoch_idx))

    return []

# ── render ────────────────────────────────────────────────────────────────
class TqdmWriter(animation.FFMpegWriter):
    def __init__(self, *args, frames: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = frames
        self._tqdm  = None

    def grab_frame(self, **savefig_kwargs):
        if self._tqdm is not None:
            self._tqdm.update(1)
        super().grab_frame(**savefig_kwargs)

    def setup(self, fig, outfile, dpi, *args, **kwargs):
        self._tqdm = tqdm(total=self.frames, desc="Rendering frames", unit="frame")
        super().setup(fig, outfile, dpi, *args, **kwargs)

    def finish(self):
        if self._tqdm is not None:
            self._tqdm.close()
        super().finish()

print(f"Total frames: {TOTAL_FRAMES} at {FPS}fps = {TOTAL_FRAMES/FPS:.1f}s")

anim = animation.FuncAnimation(
    fig, update, frames=TOTAL_FRAMES, interval=1000 / FPS, blit=False,
)
writer = TqdmWriter(
    fps=FPS, codec="libx264", bitrate=4000,
    extra_args=["-pix_fmt", "yuv420p", "-preset", "medium", "-movflags", "+faststart"],
    frames=TOTAL_FRAMES,
)

out_path = "./bert_training.mp4"
anim.save(out_path, writer=writer, dpi=DPI, savefig_kwargs={"facecolor": BG})
print(f"saved: {out_path}")
