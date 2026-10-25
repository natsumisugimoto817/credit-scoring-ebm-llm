"""Local views of an EBM for one applicant.

- plot_local_contributions: why is this applicant risky? (top risk-increasing terms)
- plot_shape_with_point: one shape function, with the applicant marked by a star
- plot_interaction_with_point: one pairwise heatmap, with the applicant marked by a star
- plot_case_report: local contributions on top, top-3 shapes below
- plot_case_view / case_view_widget: local contributions on the left, one term on the right
  (chosen from a drop-down)

Feature names are shown with a `labels` dict (feature -> business name). Missing keys fall back
to the raw feature name, so the functions work for any dataset.
"""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from ebm_utils import _axis_info, plot_interaction

RISK_UP = "#E07A73"          # soft red: risk-increasing terms
SHAPE_COLOR = "#2a78d6"      # same blue as the other EBM plots
STAR_COLOR = "#D64541"
STAR_COLOR_HEATMAP = "#222222"  # dark star on the red/blue heatmap
HEATMAP_PALENESS = 0.2          # 0 = original red/blue, 1 = white


def _pale_cmap(name="RdBu_r", paleness=HEATMAP_PALENESS, cut=0.15):
    """A softer version of a colormap: drop the darkest ends, then blend with white."""
    from matplotlib.colors import LinearSegmentedColormap
    colors = plt.get_cmap(name)(np.linspace(cut, 1 - cut, 256))
    colors[:, :3] = colors[:, :3] * (1 - paleness) + paleness
    return LinearSegmentedColormap.from_list(f"{name}_pale", colors)
HIST_COLOR = "#BDBDBD"


# ---------------------------------------------------------------- helpers
def nice(feature, labels=None):
    """Business name of a feature (falls back to the raw name)."""
    return (labels or {}).get(feature, feature)


def fmt(v):
    """Short, readable value."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (not isinstance(v, str) and pd.isna(v)):
        return "missing"
    if isinstance(v, str):
        return v
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if v == round(v):
        return f"{v:.0f}"
    return f"{v:.2f}"


def term_label(term, labels=None, width=34, sep=" &\n"):
    """'FICO score', or 'A &' / 'B' on two lines for a pair. Long names are wrapped."""
    parts = ["\n".join(textwrap.wrap(nice(f, labels), width)) for f in term.split(" & ")]
    return sep.join(parts)


def local_contributions(ebm, x_row):
    """Exact contribution of every term for one row (log-odds)."""
    return pd.Series(ebm.eval_terms(x_row)[0], index=ebm.term_names_)


def _shape(ebm, term):
    """Shape function of one main term: (kind, x, scores)."""
    cache = ebm.__dict__.setdefault("_shape_cache", {})
    if term not in cache:
        i = list(ebm.term_names_).index(term)
        d = ebm.explain_global().data(i)
        kind = ebm.feature_types_in_[list(ebm.feature_names_in_).index(term)]
        cache[term] = (kind, d["names"], np.asarray(d["scores"], dtype=float))
    return cache[term]


def shown_terms(ebm, x_row, top_n=8, min_contrib=0.01):
    """Terms that increase this applicant's risk most (at least min_contrib log-odds), largest first."""
    contrib = local_contributions(ebm, x_row)
    return list(contrib[contrib >= min_contrib].sort_values(ascending=False).head(top_n).index)


# ---------------------------------------------------------------- 1. local contributions
def plot_local_contributions(ebm, x_row, p_ref=None, labels=None, top_n=8, min_contrib=0.01, ax=None):
    """Bar chart of one applicant's risk drivers.

    Only the terms that increase the risk are shown (up to top_n). Terms below min_contrib
    (log-odds) are too small to read, so they are left out.
    p_ref: predicted probabilities of a reference population (for the percentile in the title).
    """
    contrib = local_contributions(ebm, x_row)
    p = float(ebm.predict_proba(x_row)[0, 1])

    shown = contrib[shown_terms(ebm, x_row, top_n, min_contrib)]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 0.6 * len(shown) + 1.6))
    y = np.arange(len(shown))
    ax.barh(y, shown.values, color=RISK_UP, height=0.6)
    ax.set_yticks(y, [term_label(t, labels) for t in shown.index], fontsize=10)
    for k, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_fontweight("bold" if k < 3 else "normal")      # top 3 in bold
    ax.tick_params(axis="y", pad=16, length=0)
    ax.invert_yaxis()
    if len(shown) == 0:
        ax.text(0.5, 0.5, "No feature increases the risk", transform=ax.transAxes, ha="center")
    ax.axvline(0, color="#8a8985", lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Contribution to risk (log-odds)   → riskier")

    title = f"Probability of default {p:.1%}"
    if p_ref is not None:
        p_ref = np.asarray(p_ref)
        title += f"   |   riskier than {np.mean(p_ref < p):.0%} of applicants"
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=20)
    return ax


# ---------------------------------------------------------------- 2. shape with the applicant
def plot_shape_with_point(ebm, feature, x_row, X_ref=None, labels=None, ax=None, clip=(0.01, 0.99), bins=30):
    """Shape function of one feature, with the applicant's value marked by a star.

    Continuous: shape line + grey histogram of X_ref at the bottom.
    Categorical: grey bars, star on the applicant's category.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 3.4))
    kind, names, scores = _shape(ebm, feature)
    value = x_row.iloc[0][feature]
    score = float(local_contributions(ebm, x_row)[feature])

    if kind == "continuous":
        edges = np.asarray(names, dtype=float)
        ax.step(edges, np.append(scores, scores[-1]), where="post", color=SHAPE_COLOR, lw=2)
        if X_ref is not None:
            ref = pd.to_numeric(X_ref[feature], errors="coerce").dropna()
            lo, hi = np.nanquantile(ref, clip)
            if not pd.isna(value):
                lo, hi = min(lo, value), max(hi, value)
            pad = (hi - lo) * 0.03
            ax.set_xlim(lo - pad, hi + pad)
            ax2 = ax.twinx()                         # histogram in the bottom quarter
            counts, _, _ = ax2.hist(ref.clip(lo, hi), bins=bins, color=HIST_COLOR, alpha=0.5)
            ax2.set_ylim(0, counts.max() * 4)
            ax2.set_yticks([])
            ax2.spines["right"].set_visible(False)
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
        if pd.isna(value):
            ax.text(0.02, 0.95, f"this applicant: missing ({score:+.2f})", transform=ax.transAxes,
                    va="top", color=STAR_COLOR, fontsize=9)
        else:
            ax.axvline(value, color=STAR_COLOR, ls=":", lw=1)
            ax.plot(value, score, marker="*", ms=17, color=STAR_COLOR, markeredgecolor="white", zorder=5)
            x0, x1 = ax.get_xlim()
            right = value > x0 + (x1 - x0) * 0.6          # near the right edge: write the label on the left
            ax.annotate(f"this applicant: {fmt(value)}", (value, score), xytext=(-8 if right else 8, 8),
                        textcoords="offset points", ha="right" if right else "left",
                        color=STAR_COLOR, fontsize=9)
    else:
        cats = [str(c) for c in names]
        ax.bar(cats, scores, color=HIST_COLOR)
        if str(value) in cats:
            i = cats.index(str(value))
            ax.plot(i, score, marker="*", ms=17, color=STAR_COLOR, markeredgecolor="white", zorder=5)
            ax.annotate("this applicant", (i, score), xytext=(0, 10), textcoords="offset points",
                        ha="center", color=STAR_COLOR, fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax.axhline(0, color="#8a8985", lw=1)
    ax.grid(False)
    ax.set_title(nice(feature, labels), fontsize=12, fontweight="bold", loc="center", pad=12)
    ax.set_ylabel("contribution (log-odds)", fontsize=8)
    return ax


# ---------------------------------------------------------------- 3. one figure
def top_main_terms(ebm, x_row, n=3):
    """Single-feature terms that increase this applicant's risk most (pairs are skipped)."""
    contrib = local_contributions(ebm, x_row)
    mains = contrib[[" & " not in t for t in contrib.index]]
    up = mains[mains > 0].sort_values(ascending=False)
    if len(up) < n:                                   # few risk-increasing terms: fill by size
        up = mains.reindex(mains.abs().sort_values(ascending=False).index)
    return list(up.index[:n])


def plot_case_report(ebm, x_row, X_ref=None, p_ref=None, labels=None, top_n=8, n_shapes=3, sharey=True):
    """Local contributions on top, shape functions of the top risk drivers below.

    sharey=True: the shape plots use the same y-axis, so their sizes can be compared.
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, n_shapes, height_ratios=[1.25, 1], hspace=0.45, wspace=0.35)
    plot_local_contributions(ebm, x_row, p_ref=p_ref, labels=labels, top_n=top_n, ax=fig.add_subplot(gs[0, :]))
    axes = []
    for k, f in enumerate(top_main_terms(ebm, x_row, n_shapes)):
        axes.append(plot_shape_with_point(ebm, f, x_row, X_ref=X_ref, labels=labels, ax=fig.add_subplot(gs[1, k])))
    if sharey and axes:
        lo = min(a.get_ylim()[0] for a in axes)
        hi = max(a.get_ylim()[1] for a in axes)
        for a in axes:
            a.set_ylim(lo, hi + (hi - lo) * 0.12)     # room for the "this applicant" label
    return fig


# ---------------------------------------------------------------- 4. pairwise term with the applicant
def plot_interaction_with_point(ebm, term, x_row, X_ref=None, labels=None, ax=None, clip=(0.01, 0.99)):
    """Heatmap of one pairwise term (red = adds risk, blue = reduces risk), with the applicant
    marked by a dark star. Uses plot_interaction from ebm_utils for the heatmap itself."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.8, 4.4))
    plot_interaction(ebm, term, ax=ax, X_ref=X_ref, clip=clip)
    for mesh in ax.collections:                     # softer red/blue (the colorbar follows)
        mesh.set_cmap(_pale_cmap())
    left, right = term.split(" & ")
    d = ebm.explain_global().data(list(ebm.term_names_).index(term))
    z_shape = np.asarray(d["scores"]).shape

    def position(feature, names, n_cells, axis):
        v = x_row.iloc[0][feature]
        if pd.isna(v):
            return None
        kind, ax_names = _axis_info(names, n_cells)
        if kind == "continuous":
            lo, hi = (ax.get_xlim if axis == "x" else ax.get_ylim)()
            lo, hi = min(lo, v), max(hi, v)                   # keep the star in view, with a margin
            pad = (hi - lo) * 0.04
            (ax.set_xlim if axis == "x" else ax.set_ylim)(lo - pad, hi + pad)
            return float(v)
        cats = [str(c) for c in ax_names]
        return cats.index(str(v)) if str(v) in cats else None

    xv = position(right, d["right_names"], z_shape[1], "x")
    yv = position(left, d["left_names"], z_shape[0], "y")
    score = float(local_contributions(ebm, x_row)[term])
    if xv is not None and yv is not None:
        ax.plot(xv, yv, marker="*", ms=19, color=STAR_COLOR_HEATMAP, markeredgecolor="white",
                markeredgewidth=1.5, zorder=5)
        ax.annotate(f"this applicant ({score:+.2f})", (xv, yv), xytext=(8, 8), textcoords="offset points",
                    color=STAR_COLOR_HEATMAP, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    else:
        ax.text(0.02, 0.97, f"this applicant: a value is missing ({score:+.2f})", transform=ax.transAxes,
                va="top", fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.8))
    ax.set_xlabel(nice(right, labels))
    ax.set_ylabel(nice(left, labels))
    ax.set_title(f"{nice(left, labels)}  &  {nice(right, labels)}", fontsize=12, fontweight="bold", loc="center",
                 pad=12)
    return ax


def plot_term_with_point(ebm, term, x_row, X_ref=None, labels=None, ax=None):
    """Shape (single feature) or heatmap (pair), with the applicant marked."""
    if " & " in term:
        return plot_interaction_with_point(ebm, term, x_row, X_ref=X_ref, labels=labels, ax=ax)
    return plot_shape_with_point(ebm, term, x_row, X_ref=X_ref, labels=labels, ax=ax)


# ---------------------------------------------------------------- 5. side by side, with a drop-down
def plot_case_view(ebm, x_row, term=None, X_ref=None, p_ref=None, labels=None, top_n=8, figsize=(16, 7)):
    """Local contributions on the left, one term (shape or heatmap) on the right.
    term=None: the top risk driver. For a slide, a wider figure fits better, e.g. figsize=(18, 6)."""
    terms = shown_terms(ebm, x_row, top_n)
    term = term or (terms[0] if terms else top_main_terms(ebm, x_row, 1)[0])
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
    plot_local_contributions(ebm, x_row, p_ref=p_ref, labels=labels, top_n=top_n, ax=fig.add_subplot(gs[0]))
    plot_term_with_point(ebm, term, x_row, X_ref=X_ref, labels=labels, ax=fig.add_subplot(gs[1]))
    return fig


def case_view_widget(ebm, x_row, X_ref=None, p_ref=None, labels=None, top_n=8):
    """Notebook widget: choose which term to show on the right from a drop-down."""
    from ipywidgets import Dropdown, interact

    terms = shown_terms(ebm, x_row, top_n) or top_main_terms(ebm, x_row, 3)
    options = [(term_label(t, labels, width=200, sep="  &  "), t) for t in terms]

    def show(term):
        plot_case_view(ebm, x_row, term, X_ref=X_ref, p_ref=p_ref, labels=labels, top_n=top_n)
        plt.show()

    interact(show, term=Dropdown(options=options, description="Show:", layout={"width": "480px"}))


# ---------------------------------------------------------------- 6. pair heatmap for slides (no applicant)
def plot_pair_heatmap(ebm, term, X_ref=None, labels=None, ax=None, clip=(0.01, 0.99),
                      min_count=None, hide_uncertain=False):
    """Pairwise term as a soft red/blue heatmap, with business names. No applicant marked.

    min_count: with X_ref, grey out areas with fewer rows than this (too few customers to trust).
    hide_uncertain: grey out cells whose effect can't be told apart from 0 (noise across bags).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.0))
    plot_interaction(ebm, term, ax=ax, X_ref=X_ref, clip=clip, min_count=min_count,
                     hide_uncertain=hide_uncertain)
    note = ax.get_title().split("\n", 1)[1] if "\n" in ax.get_title() else ""
    for mesh in ax.collections:
        mesh.set_cmap(_pale_cmap())
    left, right = term.split(" & ")
    ax.set_xlabel(nice(right, labels))
    ax.set_ylabel(nice(left, labels))
    ax.set_title(f"{nice(left, labels)}  &  {nice(right, labels)}", fontsize=12, fontweight="bold", pad=12)
    if note:                                          # what the grey cells mean
        ax.text(0.0, -0.32, note, transform=ax.transAxes, fontsize=9, color="#6a7179", va="top")
    return ax
