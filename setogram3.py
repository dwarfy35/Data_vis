#!/usr/bin/env python3
"""
Set'o'gram of the board game categories (Bokeh).

Implements the visualization from
  W. Freiler, K. Matkovic, H. Hauser: "Interactive Visual Analysis of Set-Typed
  Data", IEEE TVCG 14(6), 2008.

Usage
  pip install bokeh numpy pandas
  python setogram.py                       # reads bgg_data_named_na_years_removed.csv
  python setogram.py --csv my.csv --out out.html --max-card 8

How to read it (as in the paper)
  * One horizontal bar per set element (category, listed top to bottom by
    frequency) = a histogram of element frequencies. A game with several
    categories contributes to several bars.
  * Inside each bar, blocks are stacked from the left: the first block counts
    games whose set contains ONLY this element, the second block games with this
    element plus exactly one other, the third plus exactly two others, and so on.
  * Block thickness encodes the set size (blocks hang from the top edge of the
    bar). "Subtract" mode: each extra element makes the block one stripe
    thinner. "Divide" mode: thickness = full thickness / set size.
  * "Relative" mode scales all bars to the same length so blocks can be compared
    within a bar.
  * Hover over a block: it is highlighted, and the games it stands for are
    highlighted in the blocks and bars of all other categories. Hover over the
    uncovered strip of a bar to brush the whole bar. Click to pin a selection.
"""
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

WB = 0.9                       # bar width in data units (bar i is centred on x = i)
HIGHLIGHT = "#f58518"
BAR_COLOR = "#e3e7ec"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def compute(csv_path, column="category", max_card=8):
    df = pd.read_csv(csv_path, usecols=[column])
    raw = df[column].fillna("").astype(str)
    sets = [sorted({t.strip() for t in s.split(",") if t.strip()}) for s in raw]

    freq = Counter(t for s in sets for t in s)
    names = [t for t, _ in freq.most_common()]          # most frequent first
    idx = {t: i for i, t in enumerate(names)}
    n, E = len(sets), len(names)

    B = np.zeros((n, E), dtype=np.float32)              # games x elements
    for i, s in enumerate(sets):
        for t in s:
            B[i, idx[t]] = 1.0
    card = B.sum(axis=1).astype(int)

    n_empty = int((card == 0).sum())
    max_c = int(card.max())
    K = int(min(max_card, max_c))                       # number of cardinality levels
    plus = max_c > K                                    # last level means ">= K"
    level = np.minimum(card, K) - 1                     # 0-based; -1 = empty set

    counts = np.zeros((E, K), dtype=np.int64)           # games with element e and level k
    co = np.zeros((K, E, E), dtype=np.int64)            # co[k, e, f]: games with e and f at level k
    for k in range(K):
        Bk = B[level == k]
        counts[:, k] = Bk.sum(axis=0).astype(np.int64)
        co[k] = np.rint(Bk.T @ Bk).astype(np.int64)
    tot = counts.sum(axis=1)

    return dict(names=names, E=E, K=K, plus=plus, counts=counts, co=co, tot=tot,
                n_games=n, n_empty=n_empty, B=B, card=card)


def initial_geometry(d):
    """Geometry for the default view (absolute lengths, 'subtract' thickness).

    Category e is a horizontal band centred on y = E-1-e (first category on top);
    games are counted along x, blocks hang from the top edge of the band.
    """
    E, K, counts, tot = d["E"], d["K"], d["counts"], d["tot"]
    u = WB / (K + 1)
    bg = dict(left=[], right=[], bottom=[], top=[])
    blk = dict(left=[], right=[], bottom=[], top=[], color=[])
    for e in range(E):
        top = (E - 1 - e) + WB / 2
        bg["left"].append(0); bg["right"].append(int(tot[e]))
        bg["bottom"].append(top - WB); bg["top"].append(top)
        x = 0
        for k in range(K):
            c = int(counts[e, k])
            blk["left"].append(x); blk["right"].append(x + c)
            blk["bottom"].append(top - (K - k) * u); blk["top"].append(top)
            x += c
    return bg, blk


def level_colors(K, light=(198, 219, 239), dark=(8, 48, 107)):
    def hexc(t):
        return "#%02x%02x%02x" % tuple(int(round(v)) for v in t)
    if K == 1:
        return [hexc(dark)]
    return [hexc([light[j] + (dark[j] - light[j]) * k / (K - 1) for j in range(3)])
            for k in range(K)]


# ---------------------------------------------------------------------------
# Browser-side logic (BokehJS). Shared by every callback; placeholders are
# filled in by js_common().
# ---------------------------------------------------------------------------
JS_COMMON = r"""
const E = __E__, K = __K__, WB = __WB__, U = WB / (K + 1), PLUS = __PLUS__;
const names = src_names.data['name'];
const tot = src_tot.data['tot'];
const cnt = src_counts.data['c'];
const co = src_co.data['co'];
const st = src_state.data;
const bg = src_bg.data, hr = src_hlbar.data, blk = src_blk.data, hb = src_hlblk.data;

const isRel = () => rel_group.active === 1;
const isDiv = () => width_group.active === 1;
const scaleOf = (e) => isRel() ? 1.0 / tot[e] : 1.0;
const blockWidth = (k) => isDiv() ? (K * U) / (k + 1) : (K - k) * U;   // k is 0-based

function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function fmt(n) { return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ','); }

function layout() {
  let maxLen = 0;
  for (let e = 0; e < E; e++) {
    const s = scaleOf(e), top = (E - 1 - e) + WB / 2;      // first category on top
    bg['left'][e] = 0; bg['right'][e] = tot[e] * s;
    bg['bottom'][e] = top - WB; bg['top'][e] = top;
    if (tot[e] * s > maxLen) maxLen = tot[e] * s;
    let x = 0;
    for (let k = 0; k < K; k++) {
      const i = e * K + k, c = cnt[i], w = blockWidth(k);   // w = thickness of the block
      blk['left'][i] = x * s; blk['right'][i] = (x + c) * s;
      blk['bottom'][i] = top - w; blk['top'][i] = top;
      hb['left'][i] = x * s; hb['right'][i] = x * s;
      hb['bottom'][i] = top - w; hb['top'][i] = top;
      x += c;
    }
  }
  xr.start = 0; xr.end = maxLen * 1.05;
  xaxis.axis_label = isRel() ? 'share of the category\'s games (all bars scaled to equal length)'
                             : 'number of games';
}

// sel = {t: 0 none | 1 block | 2 bar, e: element, k: 0-based level}
function setHighlight(sel) {
  for (let f = 0; f < E; f++) {
    const s = scaleOf(f);
    let barH = 0;
    for (let kk = 0; kk < K; kk++) {
      const i = f * K + kk;
      let h = 0;
      if (sel.t === 1) { if (kk === sel.k) h = co[(kk * E + sel.e) * E + f]; }
      else if (sel.t === 2) { h = co[(kk * E + sel.e) * E + f]; }
      hb['right'][i] = hb['left'][i] + h * s;
      barH += h;
    }
    hr['left'][f] = 0; hr['right'][f] = barH * s;
    hr['bottom'][f] = bg['bottom'][f]; hr['top'][f] = bg['top'][f];
  }
}

const DEFAULT_INFO = '<i>Hover over a block to see which other categories the games it stands for belong to; ' +
                     'hover over the uncovered strip of a bar to brush the whole bar. ' +
                     'Click to pin the selection, click on empty space to release it.</i>';

function infoHtml(sel, pinned) {
  if (sel.t === 0) return DEFAULT_INFO;
  const e = sel.e;
  const nm = '<b>' + esc(names[e]) + '</b>' + (pinned ? ' <span style="color:#c2410c">(pinned)</span>' : '');
  const others = [];
  let head;
  if (sel.t === 1) {
    const k = sel.k, c = cnt[e * K + k];
    let what;
    if (k === 0) what = 'have <b>only</b> this category';
    else if (k === K - 1 && PLUS) what = 'have this category plus <b>at least ' + k + '</b> others';
    else what = 'have this category plus <b>exactly ' + k + '</b> other' + (k > 1 ? 's' : '');
    head = nm + ': ' + fmt(c) + ' games (' + (100 * c / tot[e]).toFixed(1) + '% of its ' + fmt(tot[e]) + ') ' + what + '.';
    for (let f = 0; f < E; f++) {
      if (f === e) continue;
      const v = co[(k * E + e) * E + f];
      if (v > 0) others.push([f, v]);
    }
  } else {
    head = nm + ': all ' + fmt(tot[e]) + ' games with this category.';
    for (let f = 0; f < E; f++) {
      if (f === e) continue;
      let v = 0;
      for (let kk = 0; kk < K; kk++) v += co[(kk * E + e) * E + f];
      if (v > 0) others.push([f, v]);
    }
  }
  others.sort((a, b) => b[1] - a[1]);
  if (others.length === 0) head += '<br>None of them has any other category.';
  else head += '<br>Most frequent other categories among these games: ' +
       others.slice(0, 10).map(o => esc(names[o[0]]) + ' (' + fmt(o[1]) + ')').join(', ') + '.';
  return head;
}

function hit(x, y) {
  const none = {t: 0, e: -1, k: -1};
  const e = (E - 1) - Math.round(y);
  if (!(e >= 0 && e < E)) return none;
  if (y < bg['bottom'][e] || y > bg['top'][e] || x < 0 || x > bg['right'][e]) return none;
  for (let k = 0; k < K; k++) {
    const i = e * K + k;
    if (blk['right'][i] > blk['left'][i] &&
        x >= blk['left'][i] && x <= blk['right'][i] &&
        y >= blk['bottom'][i] && y <= blk['top'][i]) return {t: 1, e: e, k: k};
  }
  return {t: 2, e: e, k: -1};
}

function effective() {
  if (st['pin_t'][0] !== 0) return {t: st['pin_t'][0], e: st['pin_e'][0], k: st['pin_k'][0], pinned: true};
  return {t: st['hov_t'][0], e: st['hov_e'][0], k: st['hov_k'][0], pinned: false};
}

function render(relayout) {
  if (relayout) layout();
  const sel = effective();
  setHighlight(sel);
  info_div.text = infoHtml(sel, sel.pinned);
  src_bg.change.emit(); src_blk.change.emit();
  src_hlbar.change.emit(); src_hlblk.change.emit();
}
"""

JS_WIDGET = "render(true);"

JS_MOVE = r"""
if (st['pin_t'][0] !== 0) return;
const s = hit(cb_obj.x, cb_obj.y);
if (s.t === st['hov_t'][0] && s.e === st['hov_e'][0] && s.k === st['hov_k'][0]) return;
st['hov_t'][0] = s.t; st['hov_e'][0] = s.e; st['hov_k'][0] = s.k;
render(false);
"""

JS_LEAVE = r"""
if (st['pin_t'][0] !== 0) return;
st['hov_t'][0] = 0; st['hov_e'][0] = -1; st['hov_k'][0] = -1;
render(false);
"""

JS_TAP = r"""
const s = hit(cb_obj.x, cb_obj.y);
const same = s.t === st['pin_t'][0] && s.e === st['pin_e'][0] && s.k === st['pin_k'][0];
if (s.t === 0 || same) {
  st['pin_t'][0] = 0; st['pin_e'][0] = -1; st['pin_k'][0] = -1;
  st['hov_t'][0] = 0; st['hov_e'][0] = -1; st['hov_k'][0] = -1;
} else {
  st['pin_t'][0] = s.t; st['pin_e'][0] = s.e; st['pin_k'][0] = s.k;
}
render(false);
"""


def js_common(d):
    return (JS_COMMON.replace("__E__", str(d["E"])).replace("__K__", str(d["K"]))
            .replace("__WB__", repr(WB)).replace("__PLUS__", "true" if d["plus"] else "false"))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build(d, title):
    # Bokeh is imported here so the data/JS parts can be tested without it.
    import math
    from bokeh.events import MouseLeave, MouseMove, Tap
    from bokeh.layouts import column, row
    from bokeh.models import (ColorBar, ColumnDataSource, CustomJS, Div, FixedTicker,
                              LinearColorMapper, Range1d, RadioButtonGroup)
    from bokeh.plotting import figure

    E, K = d["E"], d["K"]
    bg, blk = initial_geometry(d)
    palette = level_colors(K)
    blk["color"] = [palette[k] for _ in range(E) for k in range(K)]

    src_bg = ColumnDataSource(bg)
    src_hlbar = ColumnDataSource(dict(left=[0] * E, right=[0] * E,
                                      bottom=list(bg["bottom"]), top=list(bg["top"])))
    src_blk = ColumnDataSource(blk)
    src_hlblk = ColumnDataSource(dict(left=list(blk["left"]), right=list(blk["left"]),
                                      bottom=list(blk["bottom"]), top=list(blk["top"])))

    src_names = ColumnDataSource(dict(name=list(d["names"])))
    src_tot = ColumnDataSource(dict(tot=[int(v) for v in d["tot"]]))
    src_counts = ColumnDataSource(dict(c=[int(v) for v in d["counts"].ravel()]))
    src_co = ColumnDataSource(dict(co=[int(v) for v in d["co"].ravel()]))
    src_state = ColumnDataSource(dict(pin_t=[0], pin_e=[-1], pin_k=[-1],
                                      hov_t=[0], hov_e=[-1], hov_k=[-1]))

    max_len = float(d["tot"].max())
    width = 1150
    height = 22 * E + 160
    p = figure(width=width, height=height, title=title,
               x_range=Range1d(0, max_len * 1.05), y_range=Range1d(-0.7, E - 0.3),
               tools="pan,wheel_zoom,box_zoom,reset,save", toolbar_location="above")
    p.quad(left="left", right="right", bottom="bottom", top="top", source=src_bg,
           fill_color=BAR_COLOR, line_color=None)
    p.quad(left="left", right="right", bottom="bottom", top="top", source=src_hlbar,
           fill_color=HIGHLIGHT, line_color=None)
    p.quad(left="left", right="right", bottom="bottom", top="top", source=src_blk,
           fill_color="color", line_color="white", line_width=0.6)
    p.quad(left="left", right="right", bottom="bottom", top="top", source=src_hlblk,
           fill_color=HIGHLIGHT, line_color="white", line_width=0.6)

    p.yaxis.ticker = FixedTicker(ticks=list(range(E)))
    p.yaxis.major_label_overrides = {E - 1 - i: n for i, n in enumerate(d["names"])}
    p.yaxis.major_label_text_font_size = "9pt"
    p.ygrid.grid_line_color = None
    p.xaxis.axis_label = "number of games"
    p.xaxis.formatter.use_scientific = False

    mapper = LinearColorMapper(palette=palette, low=0.5, high=K + 0.5)
    overrides = {K: "%d+" % K} if d["plus"] else {}
    p.add_layout(ColorBar(color_mapper=mapper, ticker=FixedTicker(ticks=list(range(1, K + 1))),
                          major_label_overrides=overrides,
                          title="categories in the game's set", label_standoff=8), "right")

    width_group = RadioButtonGroup(labels=["Block width: subtract", "Block width: divide"], active=0)
    rel_group = RadioButtonGroup(labels=["Absolute heights", "Relative (equal heights)"], active=0)
    info_div = Div(text="", width=min(width, 1100), height=60,
                   styles={"font-size": "13px"})

    args = dict(src_bg=src_bg, src_hlbar=src_hlbar, src_blk=src_blk, src_hlblk=src_hlblk,
                src_names=src_names, src_tot=src_tot, src_counts=src_counts, src_co=src_co,
                src_state=src_state, width_group=width_group, rel_group=rel_group,
                info_div=info_div, xr=p.x_range, xaxis=p.xaxis[0])
    common = js_common(d)

    widget_cb = CustomJS(args=args, code=common + JS_WIDGET)
    width_group.js_on_change("active", widget_cb)
    rel_group.js_on_change("active", widget_cb)
    p.js_on_event(MouseMove, CustomJS(args=args, code=common + JS_MOVE))
    p.js_on_event(MouseLeave, CustomJS(args=args, code=common + JS_LEAVE))
    p.js_on_event(Tap, CustomJS(args=args, code=common + JS_TAP))

    info_div.text = ("<i>Hover over a block to see which other categories the games it stands for "
                     "belong to; hover over the uncovered strip of a bar to brush the whole bar. "
                     "Click to pin the selection, click on empty space to release it.</i>")
    note = ""
    if d["n_empty"]:
        note = " %d games without any category are not shown (no empty-set bar)." % d["n_empty"]
    intro = Div(text=(
        "<div style='max-width:1000px;font-size:13px'>"
        "<b>Set'o'gram</b> (Freiler, Matkovi&#263;, Hauser 2008). One bar per category = number of games "
        "with that category (a game with several categories appears in several bars). "
        "Blocks inside a bar, left to right: games with <i>only</i> this category, with exactly one other, "
        "exactly two others, ... Thinner blocks = larger sets. %d games.%s</div>"
        % (d["n_games"], note)))

    return column(intro, row(width_group, rel_group), info_div, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="bgg_data_named_na_years_removed.csv")
    ap.add_argument("--column", default="category", help="set-typed column (comma separated values)")
    ap.add_argument("--out", default="setogram_categories.html")
    ap.add_argument("--max-card", type=int, default=8,
                    help="set sizes at or above this value share the last block level (default 8)")
    args, _ = ap.parse_known_args()      # tolerate extra args when run inside Jupyter

    from bokeh.plotting import save
    from bokeh.resources import INLINE

    d = compute(args.csv, args.column, args.max_card)
    layout = build(d, "Set'o'gram of board game %s" % args.column)
    save(layout, filename=args.out, resources=INLINE, title="Set'o'gram")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
