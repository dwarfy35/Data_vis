#!/usr/bin/env python3
"""
Category adjacency matrix (Bokeh), with cells above their expected value highlighted.

Cell (i, j) = number of games that have BOTH category i and category j.

Expected value of a cell (independence / configuration-model null):
    out_i = sum over j != i of A[i, j]      (outgoing edges of category i)
    in_j  = sum over i != j of A[i, j]      (incoming edges of category j)
    total = sum of all off-diagonal entries
    expected[i, j] = out_i * in_j / total
Every game that has two categories i and j counts as one edge between them, so
out/in are the "number of edges" of a category. The matrix is symmetric, hence
out_i == in_i. Cells whose observed count exceeds the expected count are
highlighted with an orange outline.

Colour: diverging red-blue scale on log2((observed + 1) / (expected + 1)).
Blue = more games than expected ("popular" combination), red = fewer than
expected ("unpopular", including pairs that never occur together), neutral
grey-white = as expected. The +1 keeps pairs with a tiny expected count from
turning bright red just because they have no games. Its sign is identical to
the sign of (observed - expected).

Ordering: rows and columns are seriated so the matrix is as close as possible
to a Robinson matrix (values fall off monotonically when moving away from the
diagonal along every row and column): spectral ordering (Fiedler vector) as a
starting point, then a local search that minimizes the anti-Robinson deviations
  sum over i<j<k of max(0, S[i,k] - S[i,j]) + max(0, S[i,k] - S[j,k]),
where S is the colour value (log-ratio of observed to expected). Use
--order frequency to keep the categories sorted by number of games instead.

The diagonal (number of games per category) is drawn in grey and is not part
of the colour scale or the expectation.

Usage
  pip install bokeh numpy pandas
  python adjacency_matrix.py                 # reads bgg_data_named_na_years_removed.csv
  python adjacency_matrix.py --csv my.csv --out matrix.html
  python adjacency_matrix.py --effort 20     # longer search for a better order
"""
import argparse
import math
import time
from collections import Counter

import numpy as np
import pandas as pd

HIGHLIGHT = "#f58518"
DIAG_COLOR = "#9aa0a6"


# ---------------------------------------------------------------------------
# Seriation (Robinson form)
# ---------------------------------------------------------------------------
def _tri_mask(n):
    idx = np.arange(n)
    return (idx[:, None, None] < idx[None, :, None]) & (idx[None, :, None] < idx[None, None, :])


def ar_measures(S, order):
    """Anti-Robinson measures of S under `order` (higher similarity should sit nearer the diagonal).

    Returns (deviation, events, triples): summed violation size, number of violated
    triples i<j<k (S[i,k] > S[i,j] or S[i,k] > S[j,k]) and the number of triples.
    """
    n = len(order)
    M = S[np.ix_(order, order)]
    tri = _tri_mask(n)
    Mij, Mik, Mjk = M[:, :, None], M[:, None, :], M[None, :, :]
    v1, v2 = Mik - Mij, Mik - Mjk
    dev = float((np.maximum(0, v1) + np.maximum(0, v2))[tri].sum())
    events = int((((v1 > 1e-12) | (v2 > 1e-12)) & tri).sum())
    return dev, events, int(tri.sum())


def spectral_order(S):
    """Fiedler-vector ordering of the similarity matrix (Atkins et al. 1998)."""
    W = S - S.min()                       # a constant shift does not change the eigenvectors
    np.fill_diagonal(W, 0.0)
    L = np.diag(W.sum(axis=1)) - W
    vals, vecs = np.linalg.eigh(L)
    return np.argsort(vecs[:, 1], kind="stable")


def _swap_delta(S, o, p):
    """Change in anti-Robinson deviation when the neighbours at positions p, p+1 swap.

    Only triples containing both swapped items change their relative order, so this is O(n).
    """
    a, b = o[p], o[p + 1]
    Sab = S[a, b]
    d = 0.0
    if p > 0:
        c = o[:p]
        Sca, Scb = S[c, a], S[c, b]
        before = np.maximum(0, Scb - Sca) + np.maximum(0, Scb - Sab)
        after = np.maximum(0, Sca - Scb) + np.maximum(0, Sca - Sab)
        d += float((after - before).sum())
    if p + 2 < len(o):
        c = o[p + 2:]
        Sac, Sbc = S[a, c], S[b, c]
        before = np.maximum(0, Sac - Sab) + np.maximum(0, Sac - Sbc)
        after = np.maximum(0, Sbc - Sab) + np.maximum(0, Sbc - Sac)
        d += float((after - before).sum())
    return d


def _best_insertion(S, o, p):
    """Best new position (and cost change) for the item currently at position p."""
    n = len(o)
    best_delta, best_pos = 0.0, p
    w = o.copy()
    cum = 0.0
    for q in range(p, n - 1):                       # move right
        cum += _swap_delta(S, w, q)
        w[q], w[q + 1] = w[q + 1], w[q]
        if cum < best_delta - 1e-12:
            best_delta, best_pos = cum, q + 1
    w = o.copy()
    cum = 0.0
    for q in range(p, 0, -1):                       # move left
        cum += _swap_delta(S, w, q - 1)
        w[q - 1], w[q] = w[q], w[q - 1]
        if cum < best_delta - 1e-12:
            best_delta, best_pos = cum, q - 1
    return best_delta, best_pos


def _local_search(S, o, max_sweeps=50):
    o = o.copy()
    n = len(o)
    for _ in range(max_sweeps):
        improved = False
        for item in np.random.permutation(o):
            p = int(np.where(o == item)[0][0])
            delta, pos = _best_insertion(S, o, p)
            if delta < -1e-12:
                lst = list(o)
                lst.pop(p)
                lst.insert(pos, item)
                o = np.array(lst)
                improved = True
        if not improved:
            break
    return o


def seriate(S, effort=5, seed=0, verbose=True):
    """Order the rows/columns of the similarity matrix S close to Robinson form.

    Starts: spectral order and descending row-sum order, each improved by insertion local
    search; then `effort` rounds of iterated local search (random block move + local search).
    """
    n = S.shape[0]
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    base = np.arange(n)
    starts = [spectral_order(S), np.argsort(-S.sum(axis=1), kind="stable")]
    best, best_cost = None, np.inf
    for st in starts:
        o = _local_search(S, st)
        c = ar_measures(S, o)[0]
        if c < best_cost:
            best, best_cost = o, c
    for it in range(effort):
        o = best.copy()
        L = int(rng.integers(3, max(4, n // 6)))      # random block moved to a random place
        i = int(rng.integers(0, n - L))
        block = list(o[i:i + L])
        rest = list(o[:i]) + list(o[i + L:])
        j = int(rng.integers(0, len(rest) + 1))
        o = np.array(rest[:j] + block + rest[j:])
        o = _local_search(S, o)
        c = ar_measures(S, o)[0]
        if c < best_cost - 1e-12:
            best, best_cost = o, c
    if verbose:
        print("seriation: %.1fs" % (time.time() - t0))
    return best


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def compute(csv_path, column="category", order="seriation", effort=5, seed=0):
    df = pd.read_csv(csv_path, usecols=[column])
    raw = df[column].fillna("").astype(str)
    sets = [sorted({t.strip() for t in s.split(",") if t.strip()}) for s in raw]

    freq = Counter(t for s in sets for t in s)
    names = [t for t, _ in freq.most_common()]          # most frequent first
    idx = {t: i for i, t in enumerate(names)}
    n, N = len(sets), len(names)

    B = np.zeros((n, N), dtype=np.float32)              # games x categories
    for i, s in enumerate(sets):
        for t in s:
            B[i, idx[t]] = 1.0
    A = np.rint(B.T @ B).astype(np.int64)               # shared-game counts; diag = games per category

    A0 = A.copy()
    np.fill_diagonal(A0, 0)
    out_deg = A0.sum(axis=1).astype(float)              # outgoing edges per category
    in_deg = A0.sum(axis=0).astype(float)               # incoming edges per category
    total = float(A0.sum())
    expected = np.outer(out_deg, in_deg) / total
    S = np.log2((A0 + 1.0) / (expected + 1.0))          # colour value (see build())
    np.fill_diagonal(S, 0.0)

    ar_before = ar_measures(S, np.arange(N))
    if order == "seriation":
        o = seriate(S, effort=effort, seed=seed)
    else:
        o = np.arange(N)
    ar_after = ar_measures(S, o)

    names = [names[i] for i in o]
    A, A0, expected = A[np.ix_(o, o)], A0[np.ix_(o, o)], expected[np.ix_(o, o)]
    out_deg, in_deg = out_deg[o], in_deg[o]

    return dict(names=names, N=N, A=A, A0=A0, out_deg=out_deg, in_deg=in_deg,
                total=total, expected=expected, n_games=n, order=order,
                ar_before=ar_before, ar_after=ar_after)


def cell_table(d, min_ratio=1.0, min_games=1, dim=False):
    """All off-diagonal cells with a defined expected value (zero-game pairs included)."""
    names, A0, exp = d["names"], d["A0"], d["expected"]
    N = d["N"]
    rows = {k: [] for k in ("x", "y", "count", "expected", "ratio", "score", "deg_out", "deg_in")}
    for i in range(N):                                   # i = row (y), j = column (x)
        for j in range(N):
            e = float(exp[i, j])
            if i == j or e <= 0:
                continue
            c = int(A0[i, j])
            rows["x"].append(names[j]); rows["y"].append(names[i])
            rows["count"].append(c); rows["expected"].append(e)
            rows["ratio"].append(c / e)
            rows["score"].append(math.log2((c + 1) / (e + 1)))
            rows["deg_out"].append(int(d["out_deg"][i])); rows["deg_in"].append(int(d["in_deg"][j]))
    above = [(r > min_ratio) and (c >= min_games) for r, c in zip(rows["ratio"], rows["count"])]
    rows["alpha"] = [1.0 if (a or not dim) else 0.25 for a in above]
    rows["lc"] = [HIGHLIGHT if a else "#ffffff" for a in above]
    rows["lw"] = [1.6 if a else 0.3 for a in above]
    return rows, sum(above)


# ColorBrewer RdBu (11 classes), red (low) -> neutral grey-white -> blue (high)
RDBU = ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7",
        "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"]


def diverging_palette(n=256):
    """Linearly interpolated RdBu palette with n colours (index 0 = red end)."""
    stops = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in RDBU]
    out = []
    for k in range(n):
        pos = k / (n - 1) * (len(stops) - 1)
        a = min(int(pos), len(stops) - 2)
        f = pos - a
        rgb = [stops[a][j] + (stops[a + 1][j] - stops[a][j]) * f for j in range(3)]
        out.append("#%02x%02x%02x" % tuple(int(round(v)) for v in rgb))
    return out


def score_limit(scores):
    """Symmetric colour-scale limit (log2 units): 99th percentile of |score|, rounded up."""
    return max(1, int(math.ceil(float(np.percentile(np.abs(scores), 99)))))


def ratio_label(v):
    if v == 0:
        return "expected"
    return ("%g\u00d7" % 2 ** v) if v > 0 else ("1/%g\u00d7" % 2 ** -v)


# ---------------------------------------------------------------------------
# Browser-side callback: re-evaluates "above expected" from the widgets
# ---------------------------------------------------------------------------
JS_UPDATE = r"""
const d = src.data;
const minRatio = ratio_sl.value, minGames = games_sl.value;
const dim = dim_cb.active.includes(0);
let n = 0;
for (let i = 0; i < d['count'].length; i++) {
  const above = d['ratio'][i] > minRatio && d['count'][i] >= minGames;
  d['alpha'][i] = (above || !dim) ? 1.0 : 0.25;
  d['lc'][i] = above ? '%(HL)s' : '#ffffff';
  d['lw'][i] = above ? 1.6 : 0.3;
  if (above) n++;
}
src.change.emit();
info.text = '<b>' + n.toLocaleString('en-US') + '</b> of ' + d['count'].length.toLocaleString('en-US') +
  ' off-diagonal cells are above expected (observed/expected &gt; ' +
  minRatio.toFixed(1) + ' and at least ' + minGames + ' game' + (minGames === 1 ? '' : 's') + ').';
""" % {"HL": HIGHLIGHT}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build(d, title):
    from bokeh.layouts import column, row
    from bokeh.models import (CheckboxGroup, ColorBar, ColumnDataSource, CustomJS, Div,
                              FactorRange, FixedTicker, HoverTool, LinearColorMapper, Slider)
    from bokeh.plotting import figure

    names, N = d["names"], d["N"]
    cells, n_above = cell_table(d)
    src = ColumnDataSource(cells)
    L = score_limit(cells["score"])
    diag = ColumnDataSource(dict(x=names, y=names, count=[int(v) for v in np.diag(d["A"])]))

    mapper = LinearColorMapper(palette=diverging_palette(256), low=-L, high=L)

    size = 13 * N + 300
    p = figure(width=size, height=size, title=title,
               x_range=FactorRange(*names), y_range=FactorRange(*names[::-1]),
               tools="pan,wheel_zoom,box_zoom,reset,save", toolbar_location="above")
    r_main = p.rect(x="x", y="y", width=1, height=1, source=src,
                    fill_color={"field": "score", "transform": mapper},
                    fill_alpha="alpha", line_color="lc", line_width="lw")
    r_diag = p.rect(x="x", y="y", width=1, height=1, source=diag,
                    fill_color=DIAG_COLOR, line_color="white", line_width=0.3)

    p.add_tools(
        HoverTool(renderers=[r_main], tooltips=[
            ("Row category", "@y"), ("Column category", "@x"),
            ("Games sharing both", "@count{0,0}"),
            ("Expected", "@expected{0,0.00}"),
            ("Observed / expected", "@ratio{0.00}"),
            ("Row out-edges / column in-edges", "@deg_out{0,0} / @deg_in{0,0}")]),
        HoverTool(renderers=[r_diag], tooltips=[
            ("Category", "@y"), ("Games with this category", "@count{0,0}")]))

    p.xaxis.major_label_orientation = math.pi / 2
    p.xaxis.major_label_text_font_size = "7pt"
    p.yaxis.major_label_text_font_size = "7pt"
    p.axis.major_tick_line_color = None
    p.axis.axis_line_color = None
    p.grid.grid_line_color = None
    p.outline_line_color = None
    p.xaxis.axis_label = "category"
    p.yaxis.axis_label = "category"

    step = 1 if L <= 4 else 2
    ticks = [v for v in range(-L, L + 1) if v % step == 0]
    p.add_layout(ColorBar(color_mapper=mapper, ticker=FixedTicker(ticks=ticks),
                          major_label_overrides={v: ratio_label(v) for v in ticks},
                          title="observed vs expected games (log scale)",
                          label_standoff=8), "right")

    ratio_sl = Slider(start=1.0, end=10.0, value=1.0, step=0.1, width=280,
                      title="Minimum observed / expected ratio")
    games_sl = Slider(start=1, end=min(300, max(cells["count"])), value=1, step=1, width=280,
                      title="Minimum number of shared games")
    dim_cb = CheckboxGroup(labels=["Dim cells that are not above expected"], active=[])
    info = Div(width=900, styles={"font-size": "13px"}, text=(
        "<b>%s</b> of %s off-diagonal cells are above expected "
        "(observed/expected &gt; 1.0 and at least 1 game)."
        % (format(n_above, ","), format(len(cells["count"]), ","))))

    cb = CustomJS(args=dict(src=src, ratio_sl=ratio_sl, games_sl=games_sl, dim_cb=dim_cb, info=info),
                  code=JS_UPDATE)
    ratio_sl.js_on_change("value", cb)
    games_sl.js_on_change("value", cb)
    dim_cb.js_on_change("active", cb)

    if d["order"] == "seriation":
        dev0, ev0, tr = d["ar_before"]
        dev1, ev1, _ = d["ar_after"]
        order_text = (
            "<b>Order:</b> rows and columns are seriated to bring the matrix as close as possible to Robinson "
            "form, i.e. cells should fade steadily with distance from the diagonal. The size of the departures "
            "from that pattern is %.0f%% smaller than with categories sorted by number of games, but the data "
            "is not perfectly Robinsonian: %.1f%% of category triples still break the pattern (%.1f%% when "
            "sorted by number of games)."
            % (100.0 * (1 - dev1 / dev0), 100.0 * ev1 / tr, 100.0 * ev0 / tr))
    else:
        order_text = "<b>Order:</b> categories sorted by number of games."
    intro = Div(width=1000, styles={"font-size": "13px"}, text=(
        "<b>Category adjacency matrix.</b> Each cell is the number of games that have both the row "
        "and the column category (see the tooltip); the grey diagonal is the number of games per category. "
        + order_text + " "
        "<b>Colour</b> compares the observed count with the <b>expected</b> one, where expected = "
        "(out-edges of the row category &times; in-edges of the column category) / total edges and each game "
        "with two categories counts as one edge between them: "
        "<span style='color:#2166ac'><b>blue</b></span> = more games than expected (popular combination), "
        "<span style='color:#b2182b'><b>red</b></span> = fewer than expected (unpopular, down to pairs that "
        "never occur together), grey-white = as expected. "
        "Cells above expected also get an orange outline; raise the sliders to outline only the strongest, "
        "best-supported ones."))

    return column(intro, row(ratio_sl, games_sl, dim_cb), info, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="bgg_data_named_na_years_removed.csv")
    ap.add_argument("--column", default="category", help="set-typed column (comma separated values)")
    ap.add_argument("--out", default="adjacency_matrix_categories.html")
    ap.add_argument("--order", choices=["seriation", "frequency"], default="seriation",
                    help="row/column order: Robinson-like seriation (default) or by game count")
    ap.add_argument("--effort", type=int, default=5,
                    help="extra rounds of iterated local search for the seriation (default 5; more = slower, better)")
    args, _ = ap.parse_known_args()       # tolerate extra args when run inside Jupyter

    from bokeh.plotting import save
    from bokeh.resources import INLINE

    d = compute(args.csv, args.column, order=args.order, effort=args.effort)
    layout = build(d, "Shared games between board game %s" % args.column)
    save(layout, filename=args.out, resources=INLINE, title="Adjacency matrix")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
