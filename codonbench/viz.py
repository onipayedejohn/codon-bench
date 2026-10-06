"""Visual building blocks: HTML for sequences, Plotly figures for charts.

Colour roles (validated for colour-vision deficiency with the dataviz validator,
all-pairs, light surface; letters are always printed, so colour never carries
meaning alone):
  bases     A #1baf7a  C #2a78d6  G #eda100  T #4a3aa7  (U #e34948 replaces T in RNA)
  aa class  nonpolar #eda100  polar #4a3aa7  acidic #e34948  basic #2a78d6  stop #3b4252
"""

from __future__ import annotations

import html
import math

import plotly.graph_objects as go

from . import seq as S

BASE_COLORS = {"A": "#1baf7a", "C": "#2a78d6", "G": "#eda100", "T": "#4a3aa7", "U": "#e34948"}
BASE_INK = {"A": "#10241c", "C": "#ffffff", "G": "#2a1d00", "T": "#ffffff", "U": "#ffffff"}
CLASS_COLORS = {"Nonpolar": "#eda100", "Polar": "#4a3aa7", "Acidic": "#e34948", "Basic": "#2a78d6",
                "Stop": "#3b4252", "Unknown": "#9aa1ad"}
CLASS_INK = {"Nonpolar": "#2a1d00", "Polar": "#ffffff", "Acidic": "#ffffff", "Basic": "#ffffff", "Stop": "#ffffff",
             "Unknown": "#172033"}
INK, INK2, MUTED, LINE, SURFACE = "#172033", "#4b5568", "#6b7385", "#dde3ec", "#ffffff"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"


def esc(s) -> str:
    return html.escape(str(s))


# ---------------------------------------------------------------------------------------------
# Sequence HTML
# ---------------------------------------------------------------------------------------------
def base_chip(b: str, extra: str = "") -> str:
    col = BASE_COLORS.get(b)
    if not col:
        return f'<span class="chip chip-amb" title="Ambiguous base {esc(b)}">{esc(b)}</span>'
    return f'<span class="chip b-{b}" title="{S.BASES[b]["name"]}{extra}">{b}</span>'


def aa_chip(a: str, pos: int | None = None) -> str:
    a3, name, cls = S.AMINO_ACIDS.get(a, ("Xaa", "Unknown", "Unknown"))
    t = f"{name} ({a3}){f', residue {pos}' if pos else ''}: {cls.lower()}"
    return f'<span class="chip chip-aa c-{cls}" title="{esc(t)}">{esc(a)}</span>'


def bonds(a: str, b: str) -> str:
    pair = {frozenset("AT"): 2, frozenset("AU"): 2, frozenset("GC"): 3}.get(frozenset((a, b)), 0)
    if pair == 0:
        return '<span class="bond bond-none" title="No standard pair"></span>'
    t = f"{a}-{b}: {pair} hydrogen bonds"
    return f'<span class="bond" title="{t}">' + "".join('<i></i>' for _ in range(pair)) + "</span>"


def ribbon(seq: str, kind: str, start: int = 1, width: int = 40) -> str:
    """Double strand (DNA), or one strand with its DNA template (RNA), or residues (protein)."""
    window = seq[start - 1:start - 1 + width]
    if not window:
        return '<p class="muted">Nothing to show at this position.</p>'
    ruler = "".join(
        f'<span class="tick">{i}</span>' if (i % 10 == 0 or i == start) else '<span class="tick"></span>'
        for i in range(start, start + len(window)))
    if kind == "Protein":
        chips = "".join(aa_chip(a, start + k) for k, a in enumerate(window))
        return (f'<div class="ribbon" role="img" aria-label="Residues {start} to {start + len(window) - 1}">'
                f'<div class="strand"><span class="end">N</span>{chips}<span class="end">C</span></div>'
                f'<div class="strand ruler"><span class="end"></span>{ruler}<span class="end"></span></div></div>')
    rna = kind == "RNA"
    top = "".join(base_chip(b) for b in window)
    comp = S.complement(window, rna=False)
    bottom = "".join(base_chip(b) for b in comp)
    ties = "".join(bonds("T" if b == "U" else b, c) for b, c in zip(window, comp))
    top_label = "mRNA" if rna else "Coding strand"
    bottom_label = "DNA template" if rna else "Template strand"
    return f'''
<div class="ribbon" role="img" aria-label="{top_label} positions {start} to {start + len(window) - 1}, paired with its complement">
  <div class="strand-label">{top_label}</div>
  <div class="strand"><span class="end">5'</span>{top}<span class="end">3'</span></div>
  <div class="strand bonds"><span class="end"></span>{ties}<span class="end"></span></div>
  <div class="strand"><span class="end">3'</span>{bottom}<span class="end">5'</span></div>
  <div class="strand-label">{bottom_label}</div>
  <div class="strand ruler"><span class="end"></span>{ruler}<span class="end"></span></div>
</div>'''


def dogma_alignment(coding: str, codons: list, focus: int, span: int = 6) -> str:
    """The same codons through each stage of the central dogma, aligned column by column."""
    if not codons:
        return '<p class="muted">No codons to show.</p>'
    k = max(1, min(focus, len(codons)))
    lo, hi = max(1, k - span), min(len(codons), k + span)
    rows = {"Coding strand 5'→3'": [], "Template strand 3'→5'": [], "mRNA 5'→3'": [],
            "tRNA anticodon 3'→5'": [], "Amino acid": []}
    for c in codons[lo - 1:hi]:
        cur = " current" if c.index == k else ""
        dna = S.to_dna(c.codon)
        tmpl = S.complement(dna)
        rows["Coding strand 5'→3'"].append(f'<td class="cod{cur}">{"".join(base_chip(b) for b in dna)}</td>')
        rows["Template strand 3'→5'"].append(f'<td class="cod{cur}">{"".join(base_chip(b) for b in tmpl)}</td>')
        rows["mRNA 5'→3'"].append(f'<td class="cod{cur}">{"".join(base_chip(b) for b in c.codon)}</td>')
        if c.aa == "*":
            rows["tRNA anticodon 3'→5'"].append(f'<td class="cod{cur}"><span class="release">release factor</span></td>')
        else:
            rows["tRNA anticodon 3'→5'"].append(
                f'<td class="cod{cur}">{"".join(base_chip(b) for b in c.anticodon_3to5)}</td>')
        rows["Amino acid"].append(f'<td class="cod{cur}">{aa_chip(c.aa, c.index)}<div class="aa3">{c.aa3}</div></td>')
    head = "".join(f'<th class="{"current" if c.index == k else ""}">{c.index}</th>' for c in codons[lo - 1:hi])
    body = "".join(f"<tr><th scope='row'>{esc(r)}</th>{''.join(v)}</tr>" for r, v in rows.items())
    c = codons[k - 1]
    note = (f"Codon {c.index} is <b>{c.codon}</b> (mRNA positions {c.start}–{c.start + 2}). "
            + (f"A tRNA with anticodon <b>3'-{c.anticodon_3to5}-5'</b> (written 5'-{c.anticodon_5to3}-3') "
               f"brings <b>{c.name}</b> ({c.aa3}, {c.aa})." if c.aa != "*" else
               "It is a stop codon: no tRNA matches it, a release factor binds and the protein is released."))
    return (f'<div class="dogma-wrap"><table class="dogma"><thead><tr><th scope="col">Codon</th>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div><p class="dogma-note">{note}</p>')


def wrap_seq(s: str, width: int = 60, block: int = 10) -> str:
    out = []
    for i in range(0, len(s), width):
        chunk = s[i:i + width]
        out.append(f"{i + 1:>7}  " + " ".join(chunk[j:j + block] for j in range(0, len(chunk), block)))
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------------------------
def _layout(fig: go.Figure, height: int = 320, **kw) -> go.Figure:
    margin = kw.pop("margin", dict(l=56, r=20, t=24, b=48))
    showlegend = kw.pop("showlegend", False)
    fig.update_layout(
        template="simple_white", height=height, margin=margin,
        font=dict(family=FONT, size=13, color=INK2), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        hoverlabel=dict(font=dict(family=FONT, size=13), bgcolor="#ffffff", bordercolor=LINE),
        showlegend=showlegend, **kw)
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickcolor=LINE, ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f6", linecolor=LINE, zeroline=False)
    return fig


def composition_bar(counts: dict, kind: str) -> go.Figure:
    bases = ["A", "C", "G", "U" if kind == "RNA" else "T"]
    total = sum(counts.get(b, 0) for b in bases) or 1
    vals = [counts.get(b, 0) for b in bases]
    fig = go.Figure(go.Bar(
        x=vals, y=[S.BASES[b]["name"] for b in bases], orientation="h",
        marker=dict(color=[BASE_COLORS[b] for b in bases], line=dict(color=SURFACE, width=2)),
        text=[f"{v:,} ({100 * v / total:.1f}%)" for v in vals], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK2),
        hovertemplate="%{y}: %{x:,} bases<extra></extra>"))
    _layout(fig, 230, bargap=0.35)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="#eef1f6", title="Count", range=[0, max(vals) * 1.25 or 1])
    return fig


def gc_window(xs, ys, window: int, mean_gc: float) -> go.Figure:
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#2a78d6", width=2),
                               hovertemplate="Centre %{x:,}: GC %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=mean_gc, line=dict(color=MUTED, width=1, dash="dot"))
    _layout(fig, 260, hovermode="x unified")
    fig.update_xaxes(title=f"Position (window of {window} bases)")
    fig.update_yaxes(title="GC content (%)", range=[0, 100])
    return fig


def orf_map(orfs: list, length: int, selected: int | None = None) -> go.Figure:
    frames = ["+1", "+2", "+3", "-1", "-2", "-3"]
    fig = go.Figure()
    for k, o in enumerate(orfs[:300]):
        y = frames.index(o.frame)
        col = "#2a78d6" if o.frame.startswith("+") else "#4a3aa7"
        width = 16 if k == selected else 10
        fig.add_trace(go.Scatter(
            x=[o.start, o.end], y=[y, y], mode="lines", line=dict(color=col, width=width),
            hovertemplate=(f"ORF {k + 1}, frame {o.frame}<br>{o.start:,}–{o.end:,} ({o.length_nt:,} nt)<br>"
                           f"{o.length_aa} amino acids<extra></extra>")))
    _layout(fig, 280)
    fig.update_yaxes(tickvals=list(range(6)), ticktext=[f"Frame {f}" for f in frames], range=[5.6, -0.6],
                     showgrid=False)
    fig.update_xaxes(title="Position on the given strand", range=[0, length + 1], showgrid=True, gridcolor="#eef1f6")
    return fig


def aa_composition(comp: dict) -> go.Figure:
    order = sorted(comp, key=lambda a: (["Nonpolar", "Polar", "Acidic", "Basic"].index(S.AMINO_ACIDS[a][2]), a))
    total = sum(comp.values()) or 1
    fig = go.Figure(go.Bar(
        x=[f"{S.AMINO_ACIDS[a][0]}" for a in order], y=[100 * comp[a] / total for a in order],
        marker=dict(color=[CLASS_COLORS[S.AMINO_ACIDS[a][2]] for a in order], line=dict(color=SURFACE, width=2)),
        customdata=[[S.AMINO_ACIDS[a][1], S.AMINO_ACIDS[a][2], comp[a]] for a in order],
        hovertemplate="%{customdata[0]} (%{x}), %{customdata[1]}<br>%{customdata[2]} residues, %{y:.1f}%<extra></extra>"))
    _layout(fig, 300, bargap=0.25)
    fig.update_yaxes(title="Share of residues (%)")
    return fig


def hydropathy_plot(xs, ys, window: int) -> go.Figure:
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#4a3aa7", width=2),
                               hovertemplate="Residue %{x}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.add_hline(y=1.6, line=dict(color="#e34948", width=1, dash="dot"),
                  annotation_text="1.6: possible membrane-spanning stretch", annotation_font_color=MUTED,
                  annotation_position="top left")
    _layout(fig, 260, hovermode="x unified")
    fig.update_xaxes(title=f"Residue (window of {window})")
    fig.update_yaxes(title="Kyte-Doolittle hydropathy")
    return fig


def codon_wheel(table_id: int = 1, highlight: set[str] | None = None) -> go.Figure:
    """Sunburst: first base in the centre, then second, then third, with the amino acid outside."""
    tab = S.codon_table(table_id)
    order = "GACU"   # plotly lays sectors out counterclockwise, so reversed order reads U, C, A, G clockwise
    ids, labels, parents, colors, hover, values, inks = [], [], [], [], [], [], []
    for b1 in order:
        ids.append(b1); labels.append(b1); parents.append(""); colors.append(BASE_COLORS[b1]); values.append(16)
        hover.append(f"First base {b1}"); inks.append(BASE_INK[b1])
        for b2 in order:
            i2 = b1 + b2
            ids.append(i2); labels.append(b2); parents.append(b1); colors.append(BASE_COLORS[b2]); values.append(4)
            hover.append(f"{b1}{b2}_"); inks.append(BASE_INK[b2])
            for b3 in order:
                cod = i2 + b3
                dna = S.to_dna(cod)
                aa = "*" if dna in tab.stop_codons else tab.forward_table.get(dna, "X")
                a3, name, cls = S.AMINO_ACIDS[aa]
                start = " (start)" if dna == "ATG" or (table_id != 1 and dna in tab.start_codons) else ""
                ids.append(cod); labels.append(f"{b3} {a3 if aa != '*' else 'Stop'}"); parents.append(i2)
                col = CLASS_COLORS[cls]
                if highlight and cod not in highlight:
                    col = "#e9edf3"
                colors.append(col); values.append(1)
                inks.append(CLASS_INK[cls] if not (highlight and cod not in highlight) else MUTED)
                hover.append(f"Codon {cod}: {name}{start}<br>Anticodon 3'-{S.complement(cod, rna=True)}-5'")
    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total", sort=False,
        marker=dict(colors=colors, line=dict(color=SURFACE, width=1.5)), hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>", insidetextorientation="radial",
        insidetextfont=dict(family=FONT, color=inks), rotation=90))
    fig.update_layout(height=560, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor=SURFACE,
                      font=dict(family=FONT), hoverlabel=dict(bgcolor="#fff", font=dict(family=FONT)))
    return fig


def histogram(values, title: str, color: str = "#2a78d6", log: bool = False) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values, marker=dict(color=color, line=dict(color=SURFACE, width=1)),
                                 nbinsx=40, hovertemplate="%{x}: %{y:,} records<extra></extra>"))
    _layout(fig, 260, bargap=0.05)
    fig.update_xaxes(title=title, type="log" if log else "linear")
    fig.update_yaxes(title="Records")
    return fig


def codon_usage_grid(usage: dict, table_id: int = 1) -> str:
    """Classic 4 x 4 x 4 codon table with counts, as HTML."""
    tab = S.codon_table(table_id)
    order = "UCAG"
    total = sum(usage.values()) or 1
    mx = max(usage.values()) if usage else 1
    rows = []
    for b1 in order:
        for b3 in order:
            cells = []
            for b2 in order:
                cod = b1 + b2 + b3
                dna = S.to_dna(cod)
                aa = "*" if dna in tab.stop_codons else tab.forward_table.get(dna, "X")
                a3, name, cls = S.AMINO_ACIDS[aa]
                n = usage.get(cod, 0)
                alpha = 0.08 + 0.55 * (n / mx) if n else 0
                bg = f"rgba(42,120,214,{alpha:.2f})" if n else "transparent"
                cells.append(f'<td style="background:{bg}" title="{cod}: {esc(name)}, used {n} times '
                             f'({100 * n / total:.1f}%)"><span class="cg-cod">{cod}</span>'
                             f'<span class="cg-aa" style="border-color:var(--{ {"Nonpolar": "G", "Polar": "T", "Acidic": "U", "Basic": "C"}.get(cls, "stop")})">{a3}</span>'
                             f'<span class="cg-n">{n if n else ""}</span></td>')
            first = f'<th rowspan="4" class="cg-b1">{b1}</th>' if b3 == "U" else ""
            rows.append(f"<tr>{first}{''.join(cells)}<th class='cg-b3'>{b3}</th></tr>")
    head = "<tr><th></th>" + "".join(f"<th>{b}</th>" for b in order) + "<th></th></tr>"
    return (f'<table class="codon-grid" aria-label="Codon usage table"><caption>Rows: first base and third base. '
            f'Columns: second base. Shading: how often each codon is used.</caption><thead>{head}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def base_card(b: str) -> str:
    info = S.BASES[b]
    col = BASE_COLORS[b]
    hexagon = _poly(30, 32, 22, 6, 90)
    if info["rings"] == 2:
        # purine: a six-membered ring fused to a five-membered ring along one shared edge
        shape = (f'<polygon points="{_poly(46, 32, 22, 6, 90)}" style="fill:var(--{b});fill-opacity:.18;stroke:var(--{b});stroke-width:2.5"/>'
                 f'<polygon points="{_poly(11.85, 32, 18.7, 5, 180)}" style="fill:var(--{b});fill-opacity:.35;stroke:var(--{b});stroke-width:2.5"/>')
        svg = f'<svg viewBox="-9 0 80 64" width="80" height="64" aria-hidden="true">{shape}</svg>'
    else:
        svg = (f'<svg viewBox="0 0 64 64" width="64" height="64" aria-hidden="true"><polygon points="{hexagon}" '
               f'style="fill:var(--{b});fill-opacity:.18;stroke:var(--{b});stroke-width:2.5"/></svg>')
    partner = info["pairs_rna"] if b == "U" else info["pairs_dna"]
    return f'''
<div class="base-card">
  <div class="base-head"><span class="chip big b-{b}">{b}</span>{svg}</div>
  <h4>{info["name"]}</h4>
  <dl>
    <dt>Class</dt><dd>{info["kind"]} ({info["rings"]} ring{"s" if info["rings"] > 1 else ""})</dd>
    <dt>Formula</dt><dd>{info["formula"]}</dd>
    <dt>Found in</dt><dd>{info["found"]}</dd>
    <dt>Pairs with</dt><dd>{partner}{" (in DNA) or U (in RNA)" if b == "A" else ""}, {info["bonds"]} hydrogen bonds</dd>
  </dl>
</div>'''


def _poly(cx, cy, r, n, rot):
    pts = []
    for i in range(n):
        a = math.radians(rot + 360 * i / n)
        pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
    return " ".join(pts)


# ---------------------------------------------------------------------------------------------
# Plotly figures as plain HTML (no widget bridge, so re-renders never lose their model)
# ---------------------------------------------------------------------------------------------
_CONFIG = {"responsive": True, "displaylogo": False,
           "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
           "toImageButtonOptions": {"format": "png", "scale": 2}}


def plot_html(fig: go.Figure, label: str = "Chart") -> str:
    import json
    import uuid
    import plotly.io as pio
    pid = "p" + uuid.uuid4().hex[:10]
    spec = pio.to_json(fig, validate=False).replace("</", "<\\/")
    cfg = json.dumps(_CONFIG)
    h = int(fig.layout.height or 320)
    return (f'<div id="{pid}" class="plot" role="img" aria-label="{esc(label)}" style="min-height:{h}px"></div>'
            f'<script>(function(){{var f={spec};var go=function(){{var el=document.getElementById("{pid}");'
            f'if(!el||!window.Plotly){{return setTimeout(go,60);}}Plotly.newPlot(el,f.data,f.layout,{cfg}).then(function(){{if(window.cbTheme){{cbTheme(el);}}}});}};go();}})();'
            f'</script>')


def as_plot(fn):
    """Decorator: a function returning a Plotly figure becomes one returning HTML for render.ui."""
    import functools
    from shiny import ui as _ui

    @functools.wraps(fn)
    def inner(*a, **k):
        fig = fn(*a, **k)
        return None if fig is None else _ui.HTML(plot_html(fig))
    return inner


# ---------------------------------------------------------------------------------------------
# Mutation and ML charts
# ---------------------------------------------------------------------------------------------
SEV_COLORS = {"neutral": "#7a8494", "moderate": "#c98500", "severe": "#b4232c", "unknown": "#9aa3b2"}
DONOR_C, ACCEPTOR_C = "#2a78d6", "#eda100"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"]


def effect_bar(effects: list[str]) -> go.Figure:
    from . import mutations as M
    import re as _re
    labels = [_re.sub(r" \(combined.*$", "", e or "Not analysed") for e in effects]
    counts: dict[str, int] = {}
    for x in labels:
        counts[x] = counts.get(x, 0) + 1
    cats = sorted(counts, key=lambda c: -counts[c])[:14]
    sev = {k: v[1] for k, v in M.EFFECT_INFO.items()}
    col = [SEV_COLORS.get(sev.get(c, "unknown" if c != "Not analysed" else "neutral"), "#9aa3b2") for c in cats]
    fig = go.Figure(go.Bar(y=cats, x=[counts[c] for c in cats], orientation="h",
                           marker=dict(color=col, line=dict(color=SURFACE, width=2)),
                           text=[f"{counts[c]:,}" for c in cats], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:,}<extra></extra>"))
    _layout(fig, max(220, 32 * len(cats) + 70), bargap=0.3, barcornerradius=4, margin=dict(l=10, r=40, t=10, b=44))
    top = max(counts.values()) if counts else 1
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True)
    fig.update_xaxes(title="Variants", showgrid=True, gridcolor="#eef1f6", dtick=max(1, -(-top // 6)),
                     tickformat=",d", rangemode="tozero")
    return fig


def spectrum_bar(spectrum: dict) -> go.Figure:
    cats = list(spectrum)
    vals = [spectrum[c] for c in cats]
    total = max(1, sum(vals))
    fig = go.Figure(go.Bar(x=cats, y=vals, marker=dict(color="#0e5e63", line=dict(color=SURFACE, width=2)),
                           text=[f"{v / total:.0%}" for v in vals], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y:,} substitutions<extra></extra>"))
    _layout(fig, 280, bargap=0.35, barcornerradius=4)
    fig.update_yaxes(title="Substitutions", rangemode="tozero", tickformat=",d")
    fig.update_xaxes(title="Substitution (pyrimidine strand)")
    return fig


def count_bar(counts: dict, title: str, top: int = 15) -> go.Figure:
    items = list(counts.items())[:top]
    fig = go.Figure(go.Bar(y=[k for k, _ in items], x=[v for _, v in items], orientation="h",
                           marker=dict(color="#2a78d6", line=dict(color=SURFACE, width=2)),
                           hovertemplate="%{y}: %{x:,}<extra></extra>"))
    _layout(fig, max(200, 26 * len(items) + 70), bargap=0.3, barcornerradius=4, margin=dict(l=10, r=20, t=10, b=44))
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True)
    fig.update_xaxes(title=title, showgrid=True, gridcolor="#eef1f6", rangemode="tozero", tickformat=",d")
    return fig


def splice_profile_plot(profile: dict, var_pos: int, prefix: str = "g") -> go.Figure:
    """Scores of nearby candidate splice sites before (hollow) and after (filled) a variant."""
    fig = go.Figure()
    for kind, color, symbol in (("donor", DONOR_C, "circle"), ("acceptor", ACCEPTOR_C, "diamond")):
        pts = profile.get(kind, {})
        xs = sorted(pts)
        if not xs:
            continue
        ref = [pts[x][0] for x in xs]
        mut = [pts[x][1] for x in xs]
        for x, r, m in zip(xs, ref, mut):
            if abs(m - r) >= 0.02:
                fig.add_shape(type="line", x0=x, x1=x, y0=r, y1=m, line=dict(color=color, width=2))
        fig.add_trace(go.Scatter(x=xs, y=ref, mode="markers", name=f"{kind.capitalize()}, reference",
                                 marker=dict(symbol=f"{symbol}-open", size=10, color=color, line=dict(width=2)),
                                 hovertemplate=f"{kind} at {prefix}.%{{x:,}}: %{{y:.2f}} before<extra></extra>"))
        fig.add_trace(go.Scatter(x=xs, y=mut, mode="markers", name=f"{kind.capitalize()}, with variant",
                                 marker=dict(symbol=symbol, size=10, color=color, line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"{kind} at {prefix}.%{{x:,}}: %{{y:.2f}} after<extra></extra>"))
    fig.add_vline(x=var_pos, line=dict(color=MUTED, width=1, dash="dot"))
    _layout(fig, 300, showlegend=True, legend=dict(orientation="h", y=1.12, x=0, font=dict(size=12)),
            margin=dict(l=56, r=20, t=48, b=48))
    fig.update_yaxes(title="Splice-site probability", range=[-0.03, 1.05])
    fig.update_xaxes(title=f"Position ({prefix}.); dotted line marks the variant", tickformat=",d")
    return fig


def site_scan_plot(donors: tuple[list[int], list[float]], acceptors: tuple[list[int], list[float]],
                   thr_d: float, thr_a: float, length: int, annotated: dict | None = None) -> go.Figure:
    fig = go.Figure()
    for (xs, ys), name, color, thr in ((donors, "Donor (GT)", DONOR_C, thr_d), (acceptors, "Acceptor (AG)", ACCEPTOR_C, thr_a)):
        keep = [i for i, y in enumerate(ys) if y >= 0.05]
        fig.add_trace(go.Scatter(x=[xs[i] + 1 for i in keep], y=[ys[i] for i in keep], mode="markers", name=name,
                                 marker=dict(color=color, size=8, line=dict(color=SURFACE, width=1)),
                                 hovertemplate=f"{name} at %{{x:,}}: %{{y:.2f}}<extra></extra>"))
    if annotated:
        for kind, pos in annotated.items():
            for p in pos:
                fig.add_vline(x=p + 1, line=dict(color=LINE, width=1))
    fig.add_hline(y=thr_d, line=dict(color=DONOR_C, width=1, dash="dot"))
    fig.add_hline(y=thr_a, line=dict(color=ACCEPTOR_C, width=1, dash="dot"))
    _layout(fig, 300, showlegend=True, legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=56, r=20, t=44, b=48))
    fig.update_yaxes(title="Probability", range=[0, 1.05])
    fig.update_xaxes(title="Position in the sequence (candidates scoring 0.05 or more)", range=[0, length],
                     tickformat=",d")
    return fig


def coding_track_plot(xs: list[int], ys, thr: float, length: int) -> go.Figure:
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#1baf7a", width=2), name="Coding probability",
                               fill="tozeroy", fillcolor="rgba(27,175,122,0.12)",
                               hovertemplate="Window centred at %{x:,}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=thr, line=dict(color=MUTED, width=1, dash="dot"))
    _layout(fig, 240)
    fig.update_yaxes(title="Coding probability", range=[0, 1.02])
    fig.update_xaxes(title="Position (centre of each 150-base window)", range=[0, length], tickformat=",d")
    return fig


# ---- model performance -------------------------------------------------------------------------
def roc_plot(curves: dict, names: dict) -> go.Figure:
    fig = go.Figure()
    for (key, label), color in zip(names.items(), SERIES):
        c = curves[key]["roc"]
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=label, line=dict(color=color, width=2),
                                 hovertemplate=f"{label}<br>false positive rate %{{x:.3f}}<br>true positive rate "
                                               f"%{{y:.3f}}<extra></extra>"))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=LINE, width=1, dash="dot"))
    _layout(fig, 320, showlegend=True, legend=dict(orientation="h", y=-0.28, x=0), margin=dict(l=56, r=16, t=16, b=90))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.01])
    return fig


def pr_plot(curves: dict, names: dict, prevalence: float) -> go.Figure:
    fig = go.Figure()
    for (key, label), color in zip(names.items(), SERIES):
        c = curves[key]["pr"]
        fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines", name=label,
                                 line=dict(color=color, width=2),
                                 hovertemplate=f"{label}<br>recall %{{x:.3f}}<br>precision %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=prevalence, line=dict(color=LINE, width=1, dash="dot"))
    _layout(fig, 320, showlegend=True, legend=dict(orientation="h", y=-0.28, x=0), margin=dict(l=56, r=16, t=16, b=90))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.01])
    return fig


def calibration_plot(cal: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers", name="Model",
                               line=dict(color="#2a78d6", width=2), marker=dict(size=8),
                               hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<extra></extra>"))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=LINE, width=1, dash="dot"))
    _layout(fig, 300)
    fig.update_xaxes(title="Mean predicted probability (ten equal-width bins)", range=[0, 1])
    fig.update_yaxes(title="Observed fraction positive", range=[0, 1.01])
    return fig


def importance_bar(items: list[dict]) -> go.Figure:
    names = [x["feature"] for x in items][::-1]
    vals = [x["share"] * 100 for x in items][::-1]
    fig = go.Figure(go.Bar(y=names, x=vals, orientation="h", marker=dict(color="#4a3aa7"),
                           hovertemplate="%{y}: %{x:.1f}% of total gain<extra></extra>"))
    _layout(fig, 26 * len(items) + 70, bargap=0.3, barcornerradius=4, margin=dict(l=10, r=20, t=10, b=44))
    fig.update_xaxes(title="Share of total gain (%)", showgrid=True, gridcolor="#eef1f6")
    fig.update_yaxes(showgrid=False, automargin=True)
    return fig


def position_importance_plot(by_pos: dict, consensus: dict, kind: str) -> go.Figure:
    """Where the model looks (gain per position) above the information content of real sites."""
    import math
    from plotly.subplots import make_subplots
    offs = by_pos["offset"]
    info = []
    for f in consensus["freq"]:
        h = -sum(p * math.log2(p) for p in f if p > 0)
        info.append(2 - h)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Model attention: share of gain at each position",
                                        "Conservation of real sites: information content (bits)"))
    fig.add_trace(go.Bar(x=offs, y=[v * 100 for v in by_pos["share"]], marker=dict(color="#4a3aa7"),
                         hovertemplate="offset %{x}: %{y:.2f}%<extra></extra>"), row=1, col=1)
    best = ["ACGT"[max(range(4), key=lambda j: f[j])] for f in consensus["freq"]]
    labels = [b if h >= 0.3 else "" for b, h in zip(best, info)]
    fig.add_trace(go.Bar(x=offs, y=info, marker=dict(color=[BASE_COLORS[b] for b in best]), text=labels,
                         textposition="outside", textfont=dict(size=10),
                         customdata=best,
                         hovertemplate="offset %{x}: %{y:.2f} bits, most common %{customdata}<extra></extra>"), row=2, col=1)
    _layout(fig, 420, bargap=0.15, margin=dict(l=56, r=16, t=40, b=48))
    zero = "first base of the GT" if kind == "donor" else "A of the AG"
    fig.update_xaxes(title=f"Offset from the {zero}", row=2, col=1)
    fig.update_yaxes(title="%", row=1, col=1)
    fig.update_yaxes(title="bits", range=[0, 2.3], row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=INK2)
    return fig
