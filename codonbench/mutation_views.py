"""Rendering helpers for mutation results: effect cards, aligned sequences and diagrams."""
from __future__ import annotations

import pandas as pd
from shiny import ui

from . import mutations as M
from . import seq as S
from . import viz as V

SEVERITY = {"neutral": ("No change to the protein", "sev-neutral", "●"),
            "moderate": ("Changes the protein", "sev-moderate", "▲"),
            "severe": ("Likely to disrupt the protein", "sev-severe", "■"),
            "unknown": ("Effect not predicted from sequence alone", "sev-unknown", "◆")}


def _notice(kind, title, body):
    return ui.div(ui.h4(title), ui.p(body), class_=f"notice notice-{kind}", role="status")


def _facts(items):
    return ui.tags.dl(*[x for k, v in items for x in (ui.tags.dt(k), ui.tags.dd(v))], class_="facts")


def _badge(label, severity):
    name, cls, icon = SEVERITY.get(severity or "unknown", SEVERITY["unknown"])
    return f'<span class="sev {cls}" title="{name}"><span aria-hidden="true">{icon}</span> {V.esc(label)}</span>'


def _traditional(prot_change: str) -> str:
    """HBB literature numbers residues without the initiator methionine (Glu7Val is written Glu6Val)."""
    import re
    if not prot_change.startswith("p.") or "Met1" in prot_change or "=" == prot_change[-1:] and "(" in prot_change:
        return ""
    body = prot_change[2:]
    shifted = re.sub(r"(?<!fs)(?<!ext)([A-Z][a-z]{2})(\d+)", lambda m: f"{m[1]}{int(m[2]) - 1}", body)
    return "" if shifted == body else shifted


def _effect_view(ref, mut, m, e: M.Effect, is_hbb=False):
    info = [("DNA change (HGVS)", e.dna_change or "several changes"), ("Protein change (HGVS)", e.protein_change)]
    if is_hbb and _traditional(e.protein_change):
        info.append(("Traditional haemoglobin numbering", _traditional(e.protein_change)))
    if e.ref_codon:
        info.append(("Codon", f"{e.ref_codon} → {e.mut_codon}  ({S.AMINO_ACIDS[e.ref_aa][1] if e.ref_aa in S.AMINO_ACIDS else '?'}"
                              f" → {S.AMINO_ACIDS[e.mut_aa][1] if e.mut_aa in S.AMINO_ACIDS else '?'})"))
    rl, ml = len(e.ref_protein.rstrip("*")), len(e.mut_protein.rstrip("*"))
    if e.label == "Start-loss":
        info.append(("Protein length", f"{rl:,} → unknown (translation may start at a later AUG, or not at all)"))
    else:
        info.append(("Protein length", f"{rl:,} → {ml:,} amino acids" + (" (no stop codon reached)"
                                                                          if not e.mut_protein.endswith("*") else "")))
    extra = [ui.p(e.class_change, class_="mb-1")] if e.class_change else []
    extra += [ui.p(n, class_="mb-1") for n in e.notes]
    return ui.div(
        ui.h4(ui.HTML(_badge(e.label, e.severity)), class_="mb-2"),
        ui.p(e.explanation),
        *extra,
        _facts(info),
        ui.h5("DNA, reference above mutant", class_="mt-3"),
        ui.HTML(_dna_compare(ref, mut, m)),
        ui.h5("Protein, reference above mutant", class_="mt-3"),
        ui.HTML(_protein_compare(e)),
        class_="effect-card")


def _dna_compare(ref, mut, m, flank=18, prefix="c.", offset=0, codons=True):
    """Reference above mutant, with gap columns (-) so both rows stay aligned."""
    width = 2 * flank + 12
    if m is None:
        diffs = [i for i, (a, b) in enumerate(zip(ref, mut)) if a != b]
        lo = max(0, ((diffs[0] if diffs else 0) - flank) // 3 * 3)
        r_al, q_al = ref[lo:lo + width], mut[lo:lo + width]
    else:
        lo = max(0, (m.start - 1 - flank) // 3 * 3)
        hi = min(len(ref), lo + width)
        k = m.kind
        if k == "deletion":
            r_al = ref[lo:hi]
            q_al = ref[lo:m.start - 1] + "-" * (m.end - m.start + 1) + ref[m.end:hi]
        elif k in ("insertion", "duplication"):
            cut = m.start if k == "insertion" else m.end
            r_al = ref[lo:cut] + "-" * len(m.alt) + ref[cut:hi]
            q_al = ref[lo:cut] + m.alt + ref[cut:hi]
        elif k == "delins":
            n = max(len(m.ref), len(m.alt))
            r_al = ref[lo:m.start - 1] + m.ref.ljust(n, "-") + ref[m.end:hi]
            q_al = ref[lo:m.start - 1] + m.alt.ljust(n, "-") + ref[m.end:hi]
        else:
            r_al, q_al = ref[lo:hi], mut[lo:hi]
    changed = {i for i, (a, b) in enumerate(zip(r_al, q_al)) if a != b}

    def row(s, label):
        cells, ref_i = [], lo
        for i, b in enumerate(s):
            is_ref_col = r_al[i] != "-"
            if codons and is_ref_col and ref_i > lo and ref_i % 3 == 0 and i and r_al[i - 1] != "-":
                cells.append('<span class="codon-gap"></span>')
            chip = '<span class="chip chip-gap" title="gap">-</span>' if b == "-" else V.base_chip(b)
            cells.append(f'<span class="mchip{" mark" if i in changed else ""}">{chip}</span>')
            ref_i += is_ref_col
        return f'<div class="strand"><span class="end lab">{label}</span>{"".join(cells)}</div>'

    return (f'<div class="ribbon compare" aria-label="Reference and mutant DNA from {prefix}{lo + 1 + offset}">'
            f'{row(r_al, "Ref")}{row(q_al, "Mut")}'
            f'<div class="small-note" style="margin-left:34px">From {prefix}{lo + 1 + offset}. ' + ('Gaps separate codons in the reference frame. ' if codons else '') + f'Outlined bases changed; a dash marks a base that is missing in that row.</div></div>')


def _protein_compare(e: M.Effect, window=24):
    f = (e.first_diff or 1) - 1
    lo = max(0, f - 8)
    rp, mp = e.ref_protein[lo:lo + window], e.mut_protein[lo:lo + window]

    def row(p, other, label):
        cells = []
        for i, a in enumerate(p):
            diff = i >= len(other) or other[i] != a
            cells.append(f'<span class="mchip{" mark" if diff and e.first_diff else ""}">{V.aa_chip(a, lo + i + 1)}</span>')
        return f'<div class="strand"><span class="end lab">{label}</span>{"".join(cells)}</div>'

    return (f'<div class="ribbon compare">{row(rp, mp, "Ref")}{row(mp, rp, "Mut")}'
            f'<div class="small-note" style="margin-left:34px">From residue {lo + 1}. A dark square with * is a stop '
            f'codon.</div></div>')


def _fusion_chips(prot, break_a):
    k = break_a // 3
    out = []
    for i, a in enumerate(prot[:120]):
        side = "fa" if i < k else "fb"
        out.append(f'<span class="mchip {side}">{V.aa_chip(a, i + 1)}</span>')
    return f'<div class="strand" style="flex-wrap:wrap;row-gap:4px">{"".join(out)}</div>' \
           f'<div class="legend"><span><span class="sw" style="background:var(--A)"></span>from gene A</span>' \
           f'<span><span class="sw" style="background:var(--T)"></span>from gene B</span></div>'


def _chromosome(segments, total, title, marks=None, scale=None):
    """A gene drawn as a bar of segments, in HTML so it scales cleanly on any screen.

    segments: (label, start, end, colour, inverted). `scale` lets several bars share one length axis."""
    span = max(1, scale or total)
    parts = []
    for name, a, b, col, inverted in segments:
        n = b - a
        if n <= 0:
            continue
        token = {"#1baf7a": "A", "#4a3aa7": "T", "#eda100": "G", "#2a78d6": "C"}.get(col)
        ink = f"var(--{token}-ink)" if token else "#ffffff"
        col = f"var(--{token})" if token else col
        cls = "seg inv" if inverted else "seg"
        lab = ("◀ " + name) if inverted else name
        parts.append(f'<span class="{cls}" style="flex:0 0 {100 * n / span:.3f}%;background:{col};color:{ink}" '
                     f'title="{V.esc(name)}: {n:,} bases"><span class="seg-l">{V.esc(lab)}</span></span>')
    ticks = "".join(f'<span class="brk" style="left:{100 * mk / span:.3f}%"></span>' for mk in (marks or []))
    return (f'<figure class="chrom"><figcaption>{V.esc(title)} <span class="muted">({total:,} bases)</span>'
            f'</figcaption><div class="chrom-bar" role="img" aria-label="{V.esc(title)}">{"".join(parts)}{ticks}'
            f'</div></figure>')


def _effect_bar(df: pd.DataFrame):
    import plotly.graph_objects as go
    order = ["Silent", "Missense", "Nonsense", "Stop-loss", "Start-loss", "Frameshift", "In-frame insertion",
             "In-frame deletion", "Duplication", "Inversion", "No change"]
    sev = {k: v[1] for k, v in M.EFFECT_INFO.items()}
    col = {"neutral": "#7a8494", "moderate": "#c98500", "severe": "#b4232c"}
    labels = df["Effect"].astype(str).str.replace(r" \(combined.*$", "", regex=True)
    counts = labels.value_counts()
    cats = [c for c in order if c in counts] + [c for c in counts.index if c not in order and c]
    fig = go.Figure(go.Bar(
        y=cats, x=[counts[c] for c in cats], orientation="h",
        marker=dict(color=[col.get(sev.get(c, "neutral"), "#7a8494") for c in cats],
                    line=dict(color="#ffffff", width=2)),
        text=[str(counts[c]) for c in cats], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>"))
    V._layout(fig, max(220, 36 * len(cats) + 60), bargap=0.3)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    top = int(counts.max()) if len(counts) else 1
    fig.update_xaxes(title="Mutations", showgrid=True, gridcolor="#eef1f6", dtick=max(1, -(-top // 6)),
                     tickformat="d", rangemode="tozero")
    return fig


def _summary_html(df: pd.DataFrame) -> str:
    total = len(df)
    ok = int((df["Type"] != "Not analysed").sum())
    sev = {k: v[1] for k, v in M.EFFECT_INFO.items()}
    eff = df["Effect"].astype(str).str.replace(r" \(combined.*$", "", regex=True)
    severe = int(sum(sev.get(x) == "severe" for x in eff))
    neutral = int(sum(sev.get(x) == "neutral" for x in eff))
    items = [("Rows", f"{total:,}"), ("Analysed", f"{ok:,}"), ("Not analysed", f"{total - ok:,}"),
             ("Likely to disrupt the protein", f"{severe:,}"), ("Protein unchanged", f"{neutral:,}")]
    return '<dl class="facts">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in items) + "</dl>" + \
        '<div class="legend mt-2">' + "".join(
            f'<span>{_badge(lbl, s)}</span>' for lbl, s in [("Silent", "neutral"), ("Missense", "moderate"),
                                                           ("Nonsense", "severe")]) + "</div>"


def _wrap(s, n=60):
    return "\n".join(s[i:i + n] for i in range(0, len(s), n))


