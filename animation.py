"""
Render BERT masked-training animation to MP4.

Design goals:
- 1920x1080, 30fps, ~25 seconds total.
- Dark presentation-ready styling matching the widget.
- Per-epoch sentences; every 4th epoch is the benchmark "milk" sentence.
- Points migrate from random scatter to cluster positions with easeOut.
- Star marker follows the current sentence's target word.
- Loss, guess, epoch pill update each frame.
"""

import math
import random
from dataclasses import dataclass


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
from matplotlib.patheffects import withStroke
import numpy as np
from tqdm import tqdm

# ---------- palette ----------
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

CLUSTER_COLORS = {
    "drinks":  "#378ADD",
    "food":    "#D85A30",
    "places":  "#1D9E75",
    "actions": "#7F77DD",
    "animals": "#0F6E56",
    "weather": "#D4537E",
}

# ---------- data ----------
CLUSTERS = [
    {"id": "drinks",  "label": "Drinks",   "cx": 1.8, "cy": 1.2,
     "words": ["milk","coffee","tea","juice","water","wine"]},
    {"id": "food",    "label": "Food",     "cx": 1.8, "cy": 3.2,
     "words": ["bread","apple","pasta","cheese","rice","cake"]},
    {"id": "places",  "label": "Places",   "cx": 4.5, "cy": 1.2,
     "words": ["store","market","park","office","school","cafe"]},
    {"id": "actions", "label": "Actions",  "cx": 4.5, "cy": 3.2,
     "words": ["buy","read","drive","eat","walk","write"]},
    {"id": "animals", "label": "Animals",  "cx": 7.2, "cy": 1.2,
     "words": ["dog","cat","bird","horse","fish","mouse"]},
    {"id": "weather", "label": "Weather",  "cx": 7.2, "cy": 3.2,
     "words": ["rain","snow","sun","wind","cloud","storm"]},
]

BENCHMARK = {
    "tokens": ["I","went","to","the","store","to","buy","some","[MASK]","."],
    "mask_idx": 8, "truth": "milk", "is_bench": True,
}
OTHERS = [
    {"tokens": ["She","drank","a","glass","of","[MASK]","this","morning","."], "mask_idx": 5, "truth": "juice"},
    {"tokens": ["The","[MASK]","chased","the","cat","across","the","yard","."], "mask_idx": 1, "truth": "dog"},
    {"tokens": ["Heavy","[MASK]","fell","throughout","the","afternoon","."], "mask_idx": 1, "truth": "rain"},
    {"tokens": ["I","love","to","[MASK]","books","on","rainy","days","."], "mask_idx": 3, "truth": "read"},
    {"tokens": ["She","baked","fresh","[MASK]","for","breakfast","."], "mask_idx": 3, "truth": "bread"},
    {"tokens": ["They","met","at","a","quiet","[MASK]","downtown","."], "mask_idx": 5, "truth": "cafe"},
    {"tokens": ["The","[MASK]","sang","outside","my","window","."], "mask_idx": 1, "truth": "bird"},
]

# Build epoch sequence: benchmark every 4 steps, 13 epochs total.
EPOCHS = []
other_idx = 0
for i in range(17):
    if i % 4 == 0:
        EPOCHS.append({**BENCHMARK, "epoch": i * 10})
    else:
        s = OTHERS[other_idx % len(OTHERS)]
        other_idx += 1
        EPOCHS.append({**s, "is_bench": False, "epoch": i * 10})

# Assign guess + loss per epoch
bench_progression = [
    ("cactus", 9.80),
    ("chair",  5.40),
    ("bread",  2.80),
    ("water",  1.20),
    ("milk",   0.25),
]
bi = 0
for ep in EPOCHS:
    if ep.get("is_bench"):
        g, l = bench_progression[min(bi, len(bench_progression)-1)]
        ep["guess"] = g; ep["loss"] = l
        bi += 1
    else:
        e = ep["epoch"]
        rng = random.Random(e)
        if e < 20:
            ep["guess"] = rng.choice(["table","yellow","green","bright","maybe","old","house"])
            ep["loss"] = round(8.8 - e * 0.04, 2)
        elif e < 60:
            cluster_of = next(c for c in CLUSTERS if ep["truth"] in c["words"])
            pool = [w for w in cluster_of["words"] if w != ep["truth"]]
            ep["guess"] = rng.choice(pool)
            ep["loss"] = round(4.2 - (e - 20) * 0.05, 2)
        else:
            ep["guess"] = ep["truth"]
            ep["loss"] = round(1.3 - (e - 60) * 0.015, 2)

# ---------- per-word start/end positions ----------
CANVAS_W, CANVAS_H = 9.0, 4.4  # data units
rng_master = random.Random(42)

all_points = []
for c in CLUSTERS:
    for w in c["words"]:
        wr = random.Random(hash((c["id"], w)) & 0xFFFFFFFF)
        angle = wr.random() * 2 * math.pi
        radius = 0.35 + wr.random() * 0.55
        end = (c["cx"] + radius * math.cos(angle),
               c["cy"] + radius * math.sin(angle))
        start = (0.4 + rng_master.random() * (CANVAS_W - 0.8),
                 0.3 + rng_master.random() * (CANVAS_H - 0.6))
        all_points.append({
            "word": w, "cluster": c["id"],
            "color": CLUSTER_COLORS[c["id"]],
            "start": start, "end": end,
        })

# ---------- figure setup ----------
FIG_W, FIG_H = 19.2, 10.8  # 1920x1080 at 100dpi
DPI = 100

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)

# Coordinates are in figure fraction (0-1).
# Layout: header top, left panel (sentence + legend + narration),
#         right panel (embedding space). Bottom has a benchmark-guess timeline.

# Main axes for embedding space — right side
ax_space = fig.add_axes([0.40, 0.18, 0.55, 0.68])
ax_space.set_facecolor(PANEL)
ax_space.set_xlim(0, CANVAS_W)
ax_space.set_ylim(CANVAS_H, 0)  # flip y so increasing goes down like screen coords
ax_space.set_xticks([]); ax_space.set_yticks([])
for spine in ax_space.spines.values():
    spine.set_color(BORDER); spine.set_linewidth(0.5)

# Panel label for space
fig.text(0.41, 0.875, "EMBEDDING SPACE (2D projection)",
         color=TEXT_DIM, fontsize=10, weight="bold",
         family="DejaVu Sans")

# Header
fig.text(0.04, 0.93, "BERT learns to fill in the blank",
         color=TEXT, fontsize=30, weight="bold", family="DejaVu Sans")
fig.text(0.04, 0.895, "Masked language modeling — one training example per click",
         color=TEXT_MUTE, fontsize=14, family="DejaVu Sans")

# Epoch pill (top-right) — drawn with a rounded rect and text
pill_ax = fig.add_axes([0.86, 0.905, 0.10, 0.04])
pill_ax.set_xlim(0,1); pill_ax.set_ylim(0,1); pill_ax.axis("off")
pill_patch = FancyBboxPatch(
    (0.02, 0.1), 0.96, 0.8,
    boxstyle="round,pad=0.02,rounding_size=0.4",
    linewidth=0.5, edgecolor=BORDER, facecolor=PANEL_LT,
)
pill_ax.add_patch(pill_patch)
pill_text = pill_ax.text(0.5, 0.5, "Epoch 0", ha="center", va="center",
                         color=TEXT_MUTE, fontsize=13, family="DejaVu Sans")

# Left panel: sentence card
sent_ax = fig.add_axes([0.04, 0.58, 0.33, 0.30])
sent_ax.set_xlim(0,1); sent_ax.set_ylim(0,1); sent_ax.axis("off")

sent_bg = FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0,
    boxstyle="round,pad=0.0,rounding_size=0.02",
    linewidth=0.5, edgecolor=BORDER, facecolor=PANEL,
)
sent_ax.add_patch(sent_bg)
# amber accent bar on the left
sent_ax.add_patch(Rectangle((0.0, 0.0), 0.018, 1.0, facecolor=ACCENT, edgecolor="none"))

sent_ax.text(0.05, 0.88, "TRAINING EXAMPLE", color=TEXT_DIM, fontsize=10,
             weight="bold", family="DejaVu Sans", transform=sent_ax.transAxes)

# Sentence text (dynamic) — we'll render the full line as a single text element
# using matplotlib's TeX-free approach: render each line with separate colors
# via multiple text calls positioned sequentially. To keep it simple and robust,
# we render the sentence as ONE text element with the mask-and-guess inlined:
#   "I went to the store to buy some [MASK:guess] ."
# The mask chip gets colored via a single-string call where we just style the
# whole thing; a small rectangle is drawn behind the mask chunk as a highlight.

sentence_text = sent_ax.text(
    0.05, 0.65, "", color=TEXT, fontsize=17, family="DejaVu Sans",
    transform=sent_ax.transAxes, va="center", ha="left",
)
# Background highlight rect for the mask portion — drawn dynamically.
chip_bg = FancyBboxPatch((0,0), 0.01, 0.01,
                         boxstyle="round,pad=0.006,rounding_size=0.012",
                         linewidth=0.5, edgecolor=MASK_FG, facecolor=MASK_BG,
                         transform=sent_ax.transAxes, zorder=1)
sent_ax.add_patch(chip_bg)
# Separate text for the mask→guess portion so we can style it distinctly.
chip_text = sent_ax.text(0, 0.65, "", color=MASK_FG, fontsize=15, weight="bold",
                         family="DejaVu Sans Mono", transform=sent_ax.transAxes,
                         va="center", ha="left", zorder=2)
# Text that follows the chip
post_text = sent_ax.text(0, 0.65, "", color=TEXT, fontsize=17,
                         family="DejaVu Sans", transform=sent_ax.transAxes,
                         va="center", ha="left")
# Second line (if sentence wraps)
post_line2 = sent_ax.text(0.05, 0.52, "", color=TEXT, fontsize=17,
                          family="DejaVu Sans", transform=sent_ax.transAxes,
                          va="center", ha="left")
# Unused placeholder (kept for compatibility with existing code paths below)
pre_text = sentence_text

# Guess + loss line
guess_text = sent_ax.text(
    0.05, 0.12, "", color=TEXT_MUTE, fontsize=12, family="DejaVu Sans",
    transform=sent_ax.transAxes,
)

# Legend (below sentence card)
leg_ax = fig.add_axes([0.04, 0.44, 0.33, 0.11])
leg_ax.set_xlim(0,1); leg_ax.set_ylim(0,1); leg_ax.axis("off")
leg_ax.text(0, 0.92, "SEMANTIC CLUSTERS", color=TEXT_DIM, fontsize=10,
            weight="bold", family="DejaVu Sans")
for i, c in enumerate(CLUSTERS):
    col = i % 3
    row = i // 3
    lx = 0.02 + col * 0.33
    ly = 0.55 - row * 0.30
    leg_ax.scatter([lx], [ly], s=80, c=CLUSTER_COLORS[c["id"]], zorder=5, edgecolors="none")
    leg_ax.text(lx + 0.03, ly, c["label"], color=TEXT, fontsize=12,
                family="DejaVu Sans", va="center")

# Narration panel
narr_ax = fig.add_axes([0.04, 0.20, 0.33, 0.21])
narr_ax.set_xlim(0,1); narr_ax.set_ylim(0,1); narr_ax.axis("off")
narr_bg = FancyBboxPatch(
    (0,0), 1.0, 1.0,
    boxstyle="round,pad=0.0,rounding_size=0.02",
    linewidth=0.5, edgecolor=BORDER, facecolor=PANEL_LT,
)
narr_ax.add_patch(narr_bg)
narr_ax.text(0.05, 0.82, "WHAT'S HAPPENING", color=TEXT_DIM, fontsize=10,
             weight="bold", family="DejaVu Sans")
narr_text = narr_ax.text(0.05, 0.60, "", color=TEXT, fontsize=12,
                         family="DejaVu Sans", transform=narr_ax.transAxes, va="top",
                         wrap=True)

# ---------- embedding space artists ----------
# Draw cluster "glow" halos that fade in over time.
cluster_halos = []
for c in CLUSTERS:
    halo = mpatches.Ellipse(
        (c["cx"], c["cy"]), width=2.0, height=1.6,
        facecolor=CLUSTER_COLORS[c["id"]], edgecolor="none", alpha=0.0,
    )
    ax_space.add_patch(halo)
    cluster_halos.append((halo, c))

# Dots + labels, with "milk" drawn last so it stacks on top.
dot_scatter = ax_space.scatter(
    [p["start"][0] for p in all_points],
    [p["start"][1] for p in all_points],
    s=60,
    c=[p["color"] for p in all_points],
    edgecolors="none", zorder=3,
)

dot_labels = []
for p in all_points:
    txt = ax_space.text(p["start"][0] + 0.15, p["start"][1] - 0.02,
                        p["word"], color=TEXT_MUTE, fontsize=9,
                        family="DejaVu Sans", zorder=4)
    dot_labels.append(txt)

# Star marker (proper 5-pointed star) for the current sentence's truth word
def star_vertices(cx, cy, r_outer, r_inner, n=5):
    verts = []
    for i in range(2 * n):
        angle = -math.pi / 2 + i * math.pi / n
        r = r_outer if i % 2 == 0 else r_inner
        verts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return verts

_star_initial = star_vertices(0, 0, 0.26, 0.11)
star = Polygon(_star_initial, closed=True,
               facecolor=ACCENT, edgecolor=ACCENT_DK, linewidth=1.5, zorder=6)
ax_space.add_patch(star)

# Cluster label text (top-of-halo) — only visible late in training
cluster_labels_on_map = []
for c in CLUSTERS:
    t = ax_space.text(c["cx"], c["cy"] - 0.95, c["label"].upper(),
                      color=CLUSTER_COLORS[c["id"]], fontsize=11,
                      weight="bold", family="DejaVu Sans",
                      ha="center", alpha=0.0, zorder=2)
    cluster_labels_on_map.append(t)

# ---------- easing + timeline ----------
FPS = 24
HOLD_FRAMES = 32   # ~1.3s per epoch
TRANSITION_FRAMES = 22  # ~0.9s transition

def ease_in_out(t):
    return 3*t*t - 2*t*t*t

# Total frames = initial pause + for each epoch: transition (except first) + hold
TOTAL_FRAMES = 30  # initial pause to settle visually
for i in range(len(EPOCHS)):
    if i > 0:
        TOTAL_FRAMES += TRANSITION_FRAMES
    TOTAL_FRAMES += HOLD_FRAMES
# Extra hold at the end
TOTAL_FRAMES += 40

def frame_to_state(frame):
    """Return (epoch_idx, sub_t) where sub_t in [0,1] tracks global training progress."""
    f = frame
    # initial pause
    if f < 30:
        return 0, 0.0, 0.0, "hold"   # epoch idx, transition alpha, global t, phase
    f -= 30

    # Build timeline
    cur = 0
    for i in range(len(EPOCHS)):
        if i > 0:
            # Transition in
            if f < TRANSITION_FRAMES:
                # transitioning from epoch i-1 -> epoch i
                alpha = f / TRANSITION_FRAMES
                g_t = (i - 1 + alpha) / (len(EPOCHS) - 1)
                return i, alpha, g_t, "transition"
            f -= TRANSITION_FRAMES
        # Hold on epoch i
        if f < HOLD_FRAMES:
            g_t = i / (len(EPOCHS) - 1)
            return i, 1.0, g_t, "hold"
        f -= HOLD_FRAMES
    # final frames
    return len(EPOCHS) - 1, 1.0, 1.0, "hold"

def _measure_text_width_ax(text_obj, string, pos, renderer, inv):
    """Render string in text_obj at pos, return right-edge in axes fraction."""
    text_obj.set_text(string)
    text_obj.set_position(pos)
    bbox = text_obj.get_window_extent(renderer=renderer)
    return inv.transform((bbox.x1, bbox.y1))[0]

def draw_sentence(ep, guess_word):
    """Redraw sentence with mask chip. Wraps to two lines if it wouldn't fit."""
    tokens = ep["tokens"]
    mi = ep["mask_idx"]

    pre_tokens = tokens[:mi]
    post_tokens = tokens[mi+1:]

    chip_str = f"[MASK] → {guess_word}"

    renderer = fig.canvas.get_renderer()
    inv = sent_ax.transAxes.inverted()

    LEFT = 0.05
    RIGHT_LIMIT = 0.95
    LINE1_Y = 0.70
    LINE2_Y = 0.52

    # Helper: glue tokens into a string with proper spacing around punctuation.
    def stitch(toks):
        out = ""
        for t in toks:
            if t == ".":
                out = out.rstrip() + "."
            else:
                out += t + " "
        return out

    # --- Try single-line layout first ---
    pre_str = stitch(pre_tokens)
    post_str_full = stitch(post_tokens)

    sentence_text.set_fontsize(17)
    post_text.set_fontsize(17)
    chip_text.set_fontsize(15)
    post_line2.set_fontsize(17)

    # Measure pre
    sentence_text.set_text(pre_str)
    sentence_text.set_position((LEFT, LINE1_Y))
    pre_bbox = sentence_text.get_window_extent(renderer=renderer)
    pre_right = inv.transform((pre_bbox.x1, pre_bbox.y0))[0]
    chip_left = pre_right + 0.005

    # Measure chip
    chip_text.set_text(chip_str)
    chip_text.set_position((chip_left + 0.012, LINE1_Y))
    chip_bbox = chip_text.get_window_extent(renderer=renderer)
    chip_right_text = inv.transform((chip_bbox.x1, chip_bbox.y1))[0]
    chip_w = (chip_right_text - chip_left) + 0.015
    chip_right = chip_left + chip_w
    post_left_line1 = chip_right + 0.005

    # Try putting full post on line 1
    post_text.set_text(post_str_full)
    post_text.set_position((post_left_line1, LINE1_Y))
    post_bbox = post_text.get_window_extent(renderer=renderer)
    post_right = inv.transform((post_bbox.x1, post_bbox.y0))[0]

    chip_bg.set_bounds(chip_left, LINE1_Y - 0.07, chip_w, 0.14)

    # Special case: if the only post-token is a period, render it inline
    # right after the chip on line 1 and clear line 2.
    if post_tokens == ["."]:
        post_text.set_text(".")
        post_text.set_position((post_left_line1, LINE1_Y))
        post_line2.set_text("")
        return _finalize_guess_label(ep)

    if post_right <= RIGHT_LIMIT:
        # Fits on one line — clear line 2
        post_line2.set_text("")
        return _finalize_guess_label(ep)

    # --- Two-line wrap ---
    # Progressively move trailing words from post_line1 to post_line2 until line1 fits.
    # Don't allow a split that leaves only "." on line 2 — keep period glued to prior word.
    for split in range(len(post_tokens), -1, -1):
        line2_tokens = post_tokens[split:]
        # Skip splits that orphan a lone period
        if line2_tokens == ["."]:
            continue
        line1_tokens = post_tokens[:split]
        line1_post = stitch(line1_tokens)
        line2_post = stitch(line2_tokens)

        # Measure line 1
        post_text.set_text(line1_post)
        post_text.set_position((post_left_line1, LINE1_Y))
        if line1_post.strip():
            bbox = post_text.get_window_extent(renderer=renderer)
            right_edge = inv.transform((bbox.x1, bbox.y0))[0]
        else:
            right_edge = post_left_line1

        if right_edge <= RIGHT_LIMIT:
            # This split fits on line 1; put the rest on line 2
            post_line2.set_text(line2_post.rstrip())
            post_line2.set_position((LEFT, LINE2_Y))
            # Verify line 2 also fits (if not, we need to shrink, but this is rare)
            return _finalize_guess_label(ep)

    # Fallback: put all post on line 2 if even empty line 1 doesn't work
    post_text.set_text("")
    post_line2.set_text(post_str_full.rstrip())
    post_line2.set_position((LEFT, LINE2_Y))
    return _finalize_guess_label(ep)

def _finalize_guess_label(ep):
    is_bench = ep.get("is_bench", False)
    prefix = "★ Benchmark — loss: " if is_bench else "Loss: "
    color_prefix = ACCENT if is_bench else TEXT_MUTE
    guess_text.set_text(f"{prefix}{ep['loss']:.2f}")
    guess_text.set_color(color_prefix)

def interp(a, b, t):
    return a + (b - a) * t

def update_dots_for_global_t(g_t):
    """Move all dots and labels based on global training progress."""
    eased = ease_in_out(g_t)
    offsets = []
    for i, p in enumerate(all_points):
        sx, sy = p["start"]; ex, ey = p["end"]
        x = interp(sx, ex, eased)
        y = interp(sy, ey, eased)
        offsets.append([x, y])
        dot_labels[i].set_position((x + 0.15, y - 0.02))
        # Labels brighten as they cluster
        alpha = 0.4 + 0.6 * eased
        dot_labels[i].set_alpha(alpha)
    dot_scatter.set_offsets(np.array(offsets))

    # Cluster halos fade in over last third of training
    halo_strength = max(0.0, (eased - 0.55) / 0.45)
    for halo, c in cluster_halos:
        halo.set_alpha(0.12 * halo_strength)
    for t, _ in zip(cluster_labels_on_map, CLUSTERS):
        t.set_alpha(0.75 * halo_strength)

def update_star_for_epoch(ep, g_t):
    """Star sits on the current target word's current position."""
    truth = ep["truth"]
    target = next(p for p in all_points if p["word"] == truth)
    eased = ease_in_out(g_t)
    x = interp(target["start"][0], target["end"][0], eased)
    y = interp(target["start"][1], target["end"][1], eased)
    star.set_xy(star_vertices(x, y, 0.26, 0.11))

def narration_for(ep, phase, epoch_idx):
    if ep.get("is_bench"):
        if epoch_idx == 0:
            return ("We'll use this sentence as a recurring benchmark.\n"
                    "The word 'milk' is hidden. BERT has to guess it\n"
                    "from the surrounding context.")
        if ep["guess"] == ep["truth"]:
            return ("BERT has converged on the benchmark sentence.\n"
                    "The guess is correct and loss is near zero.\n"
                    "Geometry has done the work.")
        else:
            return ("Benchmark checkpoint — same sentence as before,\n"
                    "but BERT's guess has shifted as the embedding\n"
                    "space has reorganized.")
    else:
        return ("A new training example. One word is hidden;\n"
                "BERT must predict it from context.\n"
                "Gradients nudge the geometry toward truth.")

def update(frame):
    epoch_idx, alpha, g_t, phase = frame_to_state(frame)

    # Pick which epoch to display during transition (show the "incoming" one)
    ep = EPOCHS[epoch_idx]
    draw_sentence(ep, ep["guess"])

    update_dots_for_global_t(g_t)
    update_star_for_epoch(ep, g_t)

    pill_text.set_text(f"Epoch {ep['epoch']}")
    is_bench = ep.get("is_bench", False)
    pill_patch.set_facecolor(MASK_BG if is_bench else PANEL_LT)
    pill_patch.set_edgecolor(ACCENT if is_bench else BORDER)
    pill_text.set_color(ACCENT if is_bench else TEXT_MUTE)

    narr_text.set_text(narration_for(ep, phase, epoch_idx))

    return []

# ---------- render ----------
print(f"Total frames: {TOTAL_FRAMES} at {FPS}fps = {TOTAL_FRAMES/FPS:.1f}s")

anim = animation.FuncAnimation(
    fig, update, frames=TOTAL_FRAMES, interval=1000/FPS, blit=False,
)

out_path = "./bert_training.mp4"

# --- Progress bar for frame creation ---
class TqdmWriter(animation.FFMpegWriter):
    def grab_frame(self, **savefig_kwargs):
        if hasattr(self, '_tqdm') and self._tqdm is not None:
            self._tqdm.update(1)
        super().grab_frame(**savefig_kwargs)

    def setup(self, fig, outfile, dpi, *args, **kwargs):
        self._tqdm = tqdm(total=self.frames, desc="Rendering frames", unit="frame")
        super().setup(fig, outfile, dpi, *args, **kwargs)

    def finish(self):
        if hasattr(self, '_tqdm') and self._tqdm is not None:
            self._tqdm.close()
        super().finish()

writer = TqdmWriter(
    fps=FPS, codec="libx264",
    bitrate=4000,
    extra_args=["-pix_fmt", "yuv420p", "-preset", "medium", "-movflags", "+faststart"],
    frames=TOTAL_FRAMES
)

out_path = "./bert_training.mp4"
anim.save(out_path, writer=writer, dpi=DPI, savefig_kwargs={"facecolor": BG})
print(f"saved: {out_path}")