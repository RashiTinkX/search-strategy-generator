#!/usr/bin/env python
"""
Render the determinism reports to a figure + a table view.

The old harness (test.py, lost) wrote data/determinism.png; this replaces it.
Reads the JSON reports produced by determinism_eval.py and writes:

    data/determinism_v2.png        light
    data/determinism_v2_dark.png   dark (its own palette steps, not a flip)
    data/determinism_v2.md         the same numbers as a table

    python eval/plot_report.py

Design notes: one panel per metric (small multiples) rather than one grouped bar
chart, because every metric is its own 0–1 scale and a reader compares strategies
*within* a metric. Colour encodes the strategy family only — identity is already
carried by the row labels, so colour is redundant, and every bar is directly
labelled (which is also the relief the light-mode aqua needs at 2.74:1).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Validated palette subset (dataviz reference instance, slots 1-3), all-pairs in
# both modes: node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --pairs all
THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "hybrid": "#2a78d6", "llm": "#eb6834", "mesh_only": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "hybrid": "#3987e5", "llm": "#d95926", "mesh_only": "#199e70"},
}

METRICS = [
    ("within_model_mean", "within-model", "same model, rerun"),
    ("cross_model_mean", "cross-model", "different models"),
    ("heading_jaccard_mean", "heading Jaccard", "same MeSH headings"),
    ("block_count_agreement_mean", "block-count agr.", "same no. of facets"),
    ("pmid_jaccard", "PMID Jaccard", "same papers returned"),
]


def _retrieval(mode: dict, field: str):
    vals = [c["retrieval"][field] for c in mode["per_question"] if "retrieval" in c]
    return vals or None


WANTED = [
    ("llm/v1", "llm · prompt v1 (old)", "llm"),
    ("llm/v2", "llm · prompt v2", "llm"),
    ("hybrid (no closure)", "hybrid · model picks synonyms", "hybrid"),
    ("hybrid", "hybrid · derived synonyms", "hybrid"),
    ("mesh_only", "mesh_only · no LLM", "mesh_only"),
]


def load_rows(report_path: Path) -> list[dict]:
    """Fixed strategy order: prompt A/B first, then hybrid variants, then no-LLM."""
    report = json.loads(report_path.read_text())
    rows = []
    for key, label, family in WANTED:
        m = report["by_mode"].get(key)
        if m is None:
            continue
        pj = _retrieval(m, "pmid_jaccard")
        spread = _retrieval(m, "count_spread")
        rows.append({
            "label": label, "family": family,
            **{k: m[k] for k, _, _ in METRICS if k in m},
            "pmid_jaccard": statistics.fmean(pj) if pj else None,
            "count_spread": statistics.median(spread) if spread else None,
            "n_models": len(report["models"]),
        })
    return rows


def draw(rows: list[dict], mode: str, out: Path, n_models: int) -> None:
    c = THEME[mode]
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"], "text.color": c["ink"],
    })
    fig = plt.figure(figsize=(14, 7.9))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.55], hspace=0.48, wspace=0.10,
                          left=0.155, right=0.985, top=0.775, bottom=0.085)

    labels = [r["label"] for r in rows]
    y = list(range(len(rows)))[::-1]          # first strategy at the top

    fig.text(0.012, 0.955, "Is a PubMed search strategy reproducible?",
             fontsize=20, fontweight="bold", color=c["ink"])
    fig.text(0.012, 0.913,
             f"{n_models} models × 3 questions × 3 runs · map cache OFF · strict MeSH synonyms ON.  "
             "1.00 = perfect agreement.",
             fontsize=11, color=c["ink2"])
    fig.text(0.012, 0.879,
             "Colour marks the strategy family only; every bar is labelled, so identity never rests on colour.",
             fontsize=9.5, color=c["muted"])

    for i, (key, title, sub) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, i])
        vals = [r.get(key) for r in rows]
        ax.barh(y, [v if v is not None else 0 for v in vals], height=0.62,
                color=[c[r["family"]] for r in rows], zorder=3)
        for yy, v in zip(y, vals):
            if v is None:
                ax.text(0.03, yy, "not measured", va="center", fontsize=8,
                        color=c["muted"], style="italic", zorder=4)
            else:
                ax.text(v + 0.035, yy, f"{v:.2f}", va="center", fontsize=9.5,
                        fontweight="bold", color=c["ink"], zorder=4)
        if key == "cross_model_mean":         # chance level is 1/n_models
            floor = 1 / n_models
            ax.axvline(floor, color=c["ink2"], lw=1.2, ls=(0, (4, 3)), zorder=5)
            # below the last bar, where no value label can collide with it
            ax.text(floor + 0.03, -0.58, f"chance level ({floor:.2f})", fontsize=8,
                    color=c["ink2"], va="center")
        ax.set_xlim(0, 1.32)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", ".5", "1"], fontsize=9, color=c["muted"])
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=c["ink"], pad=12,
                     loc="left")
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.5, color=c["muted"])
        ax.set_yticks(y)
        ax.set_yticklabels(labels if i == 0 else [], fontsize=10, color=c["ink2"])
        ax.xaxis.grid(True, color=c["grid"], lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(c["axis"])
        ax.tick_params(length=0)

    # ---- hit-count spread: a magnitude spanning five orders of magnitude ----
    ax = fig.add_subplot(gs[1, 0:2])
    sp = [(r, r["count_spread"]) for r in rows if r["count_spread"] is not None]
    ys = list(range(len(sp)))[::-1]
    ax.barh(ys, [max(v, 1) for _, v in sp], height=0.6,
            color=[c[r["family"]] for r, _ in sp], zorder=3)
    for yy, (_, v) in zip(ys, sp):
        ax.text(max(v, 1) * 1.25, yy, f"{v:,.0f}", va="center", fontsize=9.5,
                fontweight="bold", color=c["ink"], zorder=4)
    ax.set_xscale("symlog")
    ax.set_xlim(0, 3_000_000)
    ax.set_xticks([0, 1e2, 1e4, 1e6])
    ax.set_xticklabels(["0", "100", "10k", "1M"])
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r, _ in sp], fontsize=9.5, color=c["ink2"])
    ax.set_title("median spread in PubMed hit count across models",
                 fontsize=11.5, fontweight="bold", color=c["ink"], pad=12, loc="left")
    ax.text(0, 1.03, "lower is better · log scale", transform=ax.transAxes,
            fontsize=8.5, color=c["muted"])
    ax.xaxis.grid(True, color=c["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(c["axis"])
    ax.tick_params(length=0, labelsize=8.5, colors=c["muted"])

    # ---- the two effects that are not agreement rates ----
    tiles = [
        ("104 → 5", "MeSH headings that failed exact\nresolution, prompt v1 → v2"),
        ("7.7 → 3.8 KB", "median compiled query length\n(PubMed's URL limit is ~8 KB)"),
        ("0 · 0", "hybrid: out-of-slate ids ·\nunresolvable headings, all runs"),
    ]
    for j, (big, cap) in enumerate(tiles):
        ax = fig.add_subplot(gs[1, 2 + j])
        ax.axis("off")
        ax.text(0.0, 0.72, big, fontsize=21, fontweight="bold", color=c["ink"],
                transform=ax.transAxes)
        ax.text(0.0, 0.30, cap, fontsize=9, color=c["ink2"], transform=ax.transAxes,
                linespacing=1.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c[f]) for f in ("llm", "hybrid", "mesh_only")]
    fig.legend(handles, ["llm — model proposes freely",
                         "hybrid — model selects from a MeSH slate",
                         "mesh_only — no LLM"],
               loc="upper right", bbox_to_anchor=(0.988, 0.985), frameon=False,
               fontsize=9.5, labelcolor=c["ink2"], handlelength=1.1, handleheight=0.9)

    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  figure -> {out}")


GLOSSARY = """
## What the columns mean

| column | definition |
|---|---|
| **within-model** | One model, one question, run 3×: the share of runs that compiled the **byte-identical** query. 1.00 means rerunning your own search is safe. |
| **cross-model** | Different models, same question: the share that compiled the byte-identical query (modal share). **Chance level is 1/number-of-models**, so this number is meaningless without knowing how many models were in the run, and is not comparable across runs of different size. |
| **heading Jaccard** | Mean pairwise overlap of the chosen MeSH heading *sets* — \\|A∩B\\| / \\|A∪B\\|. Ignores wording, order and formatting, so it measures whether models agreed on the *substance*. 1.00 = same headings. |
| **block-count agr.** | Share of models that agree on how many ANDed facets the question has. Catches the "one model used 1 block, another used 7" failure directly. |
| **PMID Jaccard** | Mean pairwise overlap of the PMID sets PubMed **actually returns** for each pair of queries. 1.00 = the two searches retrieve the same papers. This is the metric a reviewer should care about: two differently-worded queries can retrieve one corpus, and two similar-looking queries can retrieve different ones. |
| **median hit spread** | Across models, the largest hit count minus the smallest, median over the questions. An absolute-count companion to PMID Jaccard: a spread of 356,496 means some model's query was wildly broader than another's. |

## What the strategies mean

| strategy | meaning |
|---|---|
| **llm** | The model proposes the concept blocks and MeSH headings freely. `prompt v1` is the original prompt, `prompt v2` the rewritten one. |
| **hybrid** | A deterministic MeSH lookup over the question produces a numbered candidate slate; the model may only *select* ids from it, so it cannot invent vocabulary. |
| **span grouping** | Who decides which headings share an OR block. The *question* decides: candidates found from overlapping question words are alternatives for one facet; candidates from disjoint words stay ANDed. The alternative (`model grouping`) lets the model group them, which let one model OR a technique with a brain region. |
| **closure** | Who picks the synonyms *inside* a chosen facet. With closure, the facet's canonical vocabulary is derived from the slate (every exact match, plus reworded matches at least as well-supported); the model only decides whether the facet belongs. Without it, the model names the synonyms — and models disagree there far more than they disagree about facets. |
| **mesh_only** | No LLM at all — every maximal MeSH match in the question becomes a block. Model-independent by construction. |

Two policies were measured and rejected: **slot merging** (merging blocks the model
gave the same PICO label — models label identical content with different slots, and
it cost 0.05 cross-model), and **model grouping** (letting the model decide which
headings share an OR block — one model ORed a technique with a brain region).

A `–` means the row was computed offline from stored selections, so it never issued
PubMed queries and has no retrieval numbers.
"""


def write_table(rows: list[dict], out: Path, n_models: int) -> None:
    """The table view: same numbers, readable without the figure."""
    head = ["strategy"] + [t for _, t, _ in METRICS] + ["median hit spread"]
    lines = [f"# Determinism results ({n_models} models × 3 questions × 3 runs, "
             f"cache OFF, strict ON)", "",
             f"Chance-level cross-model agreement at n={n_models} is "
             f"{1 / n_models:.2f}; numbers are not comparable across different "
             f"model-set sizes.", "",
             "| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        cells = [r["label"]]
        for key, _, _ in METRICS:
            v = r.get(key)
            cells.append(f"{v:.2f}" if v is not None else "–")
        cells.append(f"{r['count_spread']:,.0f}" if r["count_spread"] is not None else "–")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(GLOSSARY)
    out.write_text("\n".join(lines) + "\n")
    print(f"  table  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(ROOT / "data" / "determinism_v3.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "determinism_v3.png"))
    args = ap.parse_args()

    rows = load_rows(Path(args.report))
    n_models = rows[0]["n_models"] if rows else 0
    out = Path(args.out)
    draw(rows, "light", out, n_models)
    draw(rows, "dark", out.with_name(out.stem + "_dark" + out.suffix), n_models)
    write_table(rows, out.with_suffix(".md"), n_models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
