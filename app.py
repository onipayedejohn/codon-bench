"""Codon Bench: sequence, mutation and machine-learning analysis for DNA, RNA and proteins (Shiny for Python).

Author: Onipayede John Kwaku. Licence: MIT.

Run locally:  shiny run --reload app.py
Deploy:       Posit Connect Cloud, primary file app.py (see README).
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

from codonbench import batch as B
from codonbench import config as C
from codonbench import demos, docread, ml
from codonbench import files as F
from codonbench import seq as S
from codonbench import viz as V
from codonbench.mutations_ui import mutation_server, mutation_ui
from codonbench.performance_ui import performance_server, performance_ui

_plot = V.as_plot

ROOT = Path(__file__).parent
SAMPLES = ROOT / "data" / "samples"
SINGLE_LIMIT = 2_000_000          # letters analysed on the single-sequence page
PROGRESS: dict[str, tuple[float, str]] = {}

CODES = {str(k): f"{v} (table {k})" for k, v in S.GENETIC_CODES.items()}


def brand():
    chips = "".join(f'<span style="background:var(--{b})"></span>' for b in "ACGT")
    return ui.HTML(f'<span class="brand-mark" aria-hidden="true">{chips}</span><span class="brand-text">'
                   f'<span class="brand-name">{C.APP_NAME}</span><span class="brand-tag">Sequence, mutation and ML '
                   f'analysis</span></span>')


def footer():
    a = C.AUTHOR
    links = [f'<a href="{a["github"]}" target="_blank" rel="noopener">GitHub</a>']
    if a.get("repository"):
        links.append(f'<a href="{a["repository"]}" target="_blank" rel="noopener">Source code</a>')
    if a.get("linkedin"):
        links.append(f'<a href="{a["linkedin"]}" target="_blank" rel="noopener">LinkedIn</a>')
    if a.get("email"):
        links.append(f'<a href="mailto:{a["email"]}">{a["email"]}</a>')
    return ui.HTML(f"""
<footer class="site-footer" role="contentinfo">
  <div class="foot-grid">
    <div><p class="foot-name">{C.APP_NAME} <span class="foot-version">v{C.VERSION}</span></p>
      <p>{C.TAGLINE}.</p></div>
    <div><p class="foot-label">Designed and developed by</p>
      <p class="foot-author">{a["name"]}</p><p>{a["role"]}</p><p class="foot-links">{" ".join(links)}</p></div>
    <div><p class="foot-label">Cite this software</p><p class="foot-cite">{C.CITATION}</p></div>
  </div>
  <p class="foot-legal">© {C.RELEASED} {a["name"]}. All rights reserved. Released under the {C.LICENSE}. For research
  use; not for clinical diagnosis.</p>
</footer>""")


def log_row(kind: str, item: str, result: str) -> dict:
    return {"Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "Tool": kind,
            "Input": item[:160], "Result": result[:240]}


def notice(kind: str, title: str, body: str):
    return ui.div(ui.h4(title), ui.p(body), class_=f"notice notice-{kind}", role="status")


def facts(items: list[tuple[str, str]]):
    return ui.tags.dl(*[x for k, v in items for x in (ui.tags.dt(k), ui.tags.dd(v))], class_="facts")


# ---------------------------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------------------------
analyze_panel = ui.nav_panel(
    "Analyse",
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Your sequence"),
            ui.input_select("demo", "Start from an example", {k: v[0] for k, v in demos.SINGLE.items()} |
                            {"own": "My own sequence"}, selected="hbb"),
            ui.input_text_area("seq_text", "Paste a sequence (FASTA, GenBank or plain letters)",
                               value=demos.SINGLE["hbb"][1], rows=9, width="100%",
                               placeholder=">my gene\nATGGCC..."),
            ui.input_file("seq_file", "Or upload a file (the first sequence is used)", accept=sorted(F.SEQ_EXT),
                          multiple=False, button_label="Choose file", placeholder="FASTA, TXT, GenBank"),
            ui.input_radio_buttons("strand", "For DNA, the sequence above is the",
                                   {"coding": "coding (sense) strand", "template": "template (antisense) strand"},
                                   selected="coding"),
            ui.input_select("code", "Genetic code", CODES, selected="1"),
            ui.input_numeric("min_orf", "Shortest ORF to report (amino acids)", 30, min=5, max=5000, step=5),
            ui.input_checkbox("alt_starts", "Also allow alternative start codons (GTG, TTG ...)", False),
            ui.input_action_button("go", "Analyse sequence", class_="btn-primary w-100"),
            ui.p("Analysis runs on this server. Sequences are not saved to any database or shared.", class_="small-note"),
            width=340, open={"desktop": "open", "mobile": "always"},
        ),
        ui.output_ui("analysis_head"),
        ui.output_ui("analysis_body"),
    ),
)

genetic_code_panel = ui.nav_panel(
    "Genetic code",
    ui.h2("The genetic code, bases and anticodons", class_="page-title"),
    ui.p("Sixty-four codons, read from the centre outwards: first base, second base, third base, then the amino acid "
         "it codes for. Hover any codon for its tRNA anticodon.", class_="lede"),
    ui.layout_columns(
        ui.card(
            ui.card_header("Codon wheel"),
            ui.layout_columns(
                ui.input_select("wheel_code", "Genetic code", CODES, selected="1"),
                ui.input_text("wheel_find", "Highlight an amino acid or codon", placeholder="e.g. Leu, W, AUG, stop"),
                col_widths=(6, 6),
            ),
            ui.output_ui("aa_legend"),
            ui.output_ui("wheel"),
        ),
        ui.card(
            ui.card_header("How to read a codon and its anticodon"),
            ui.markdown(
                "- **Codon:** three bases on the mRNA, read 5'→3' by the ribosome.\n"
                "- **Anticodon:** three bases on the tRNA that pair with the codon. Pairing is antiparallel, so the "
                "anticodon for mRNA codon 5'-AUG-3' is 3'-UAC-5', which is written 5'-CAU-3'.\n"
                "- **Start:** AUG codes for methionine and also starts translation.\n"
                "- **Stop:** UAA, UAG and UGA (standard code) have no tRNA. A release factor ends translation.\n"
                "- **Wobble:** the third codon base pairs loosely with the first anticodon base, so one tRNA can read "
                "several codons. Inosine in an anticodon can pair with U, C or A.\n"
                "- **Degeneracy:** 61 sense codons code for 20 amino acids, so most amino acids have several codons. "
                "Leucine, serine and arginine have six; methionine and tryptophan have one."),
            ui.h5("Nucleotides", class_="mt-3"),
            ui.p("A nucleotide is a nitrogenous base, a five-carbon sugar and a phosphate group. DNA uses "
                 "deoxyribose and the bases A, G, C and T. RNA uses ribose and replaces T with U. Purines (A, G) "
                 "have two rings; pyrimidines (C, T, U) have one. A purine always pairs with a pyrimidine, which keeps "
                 "the double helix the same width.", class_="muted"),
        ),
        col_widths=(7, 5),
    ),
    ui.card(ui.card_header("Nitrogenous bases"), ui.HTML('<div class="base-grid">' +
                                                         "".join(V.base_card(b) for b in "AGCTU") + "</div>")),
)

batch_panel = ui.nav_panel(
    "Batch",
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Many sequences at once"),
            ui.input_file("batch_file", f"Upload a file, up to {C.MAX_UPLOAD_MB} MB", accept=F.ACCEPT_ATTR, multiple=False,
                          button_label="Choose file", placeholder="FASTA, FA, FNA, TXT, CSV, TSV, XLSX, .gz"),
            ui.p("Or try an example", class_="form-label mb-1"),
            ui.div(ui.input_action_button("b_demo_fa", "FASTA", class_="btn-ghost"),
                   ui.input_action_button("b_demo_csv", "CSV", class_="btn-ghost"),
                   ui.input_action_button("b_demo_xlsx", "Excel", class_="btn-ghost"),
                   ui.input_action_button("b_demo_bad", "Unrelated file", class_="btn-ghost"),
                   class_="demo-row"),
            ui.input_select("b_code", "Genetic code", CODES, selected="1"),
            ui.input_numeric("b_min_orf", "Shortest ORF to report (amino acids)", 30, min=5, max=5000, step=5),
            ui.markdown("Tables need a column of sequences, ideally named *sequence*. Records longer than "
                        f"{F.KEEP_LIMIT:,} letters get composition only."),
            width=340, open={"desktop": "open", "mobile": "always"},
        ),
        ui.output_ui("batch_status"),
        ui.output_ui("batch_body"),
    ),
)

document_panel = ui.nav_panel(
    "Read a document",
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Images and PDFs"),
            ui.input_file("doc_file", "Upload a photo, scan or PDF", accept=sorted(F.IMAGE_EXT | F.DOC_EXT),
                          multiple=False, button_label="Choose file", placeholder="PNG, JPG, TIFF, PDF"),
            ui.p("Or try an example", class_="form-label mb-1"),
            ui.div(ui.input_action_button("d_demo_page", "Textbook page", class_="btn-ghost"),
                   ui.input_action_button("d_demo_photo", "Phone photo", class_="btn-ghost"),
                   ui.input_action_button("d_demo_table", "Peptide table", class_="btn-ghost"),
                   ui.input_action_button("d_demo_pdf", "PDF", class_="btn-ghost"),
                   ui.input_action_button("d_demo_bad", "Unrelated poster", class_="btn-ghost"),
                   class_="demo-row"),
            ui.markdown("The app reads the text (OCR), works out which lines are headers, sequences, table rows or "
                        "ordinary text, and joins sequence lines into sequences. Check them against the page: OCR "
                        "can misread a letter."),
            width=340, open={"desktop": "open", "mobile": "always"},
        ),
        ui.output_ui("doc_status"),
        ui.output_ui("doc_body"),
    ),
)

history_panel = ui.nav_panel(
    "History",
    ui.h2("Session history", class_="page-title"),
    ui.p("Everything you analysed in this browser session, newest first. It is kept only while this page is open and "
         "is not stored on the server.", class_="lede"),
    ui.div(ui.download_button("dl_history", "Download history (CSV)", class_="btn-primary w-auto"),
           ui.input_action_button("clear_history", "Clear history", class_="btn-ghost ms-2"), class_="mb-3"),
    ui.output_ui("history_view"),
)

about_panel = ui.nav_panel(
    "About",
    ui.div(
        ui.h2(f"About {C.APP_NAME}", class_="page-title"),
        ui.markdown((ROOT / "docs" / "about.md").read_text() if (ROOT / "docs" / "about.md").exists()
                    else f"{C.APP_NAME}: {C.TAGLINE}."),
        class_="about"),
)

app_ui = ui.page_navbar(
    analyze_panel,
    mutation_ui(),
    batch_panel,
    document_panel,
    genetic_code_panel,
    performance_ui(),
    history_panel,
    about_panel,
    ui.nav_spacer(),
    ui.nav_control(ui.input_dark_mode(id="theme")),
    title=brand(),
    id="nav",
    window_title=f"{C.APP_NAME}: DNA, RNA, protein, mutation and ML analysis",
    footer=footer(),
    header=ui.head_content(
        ui.tags.link(rel="stylesheet", href="styles.css"),
        ui.tags.meta(name="author", content=C.AUTHOR["name"]),
        ui.tags.script(src="plotly.min.js"),
        ui.tags.script(src="theme.js"),
        ui.tags.meta(name="description", content="Analyse DNA, RNA and protein sequences, annotate mutations in any "
                                                 "genome and predict splice sites and coding potential with machine "
                                                 "learning."),
    ),
    fillable=False,
    lang="en",
)


# ---------------------------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------------------------
def server(input: Inputs, output: Outputs, session: Session):
    source = reactive.value(demos.SINGLE["hbb"][1])
    source_name = reactive.value("")
    history = reactive.value([])

    def log(kind: str, item: str, result: str = ""):
        with reactive.isolate():
            cur = history()
        row = log_row(kind, item, result)
        if cur and all(cur[0][k] == row[k] for k in ("Tool", "Input", "Result")):
            return
        history.set([row] + cur[:499])

    @reactive.effect
    def _log_analysis():
        p = parsed()
        if p is None or p.kind == "Unknown":
            return
        with reactive.isolate():
            name = (p.header or source_name() or "pasted sequence")[:80]
        log("Analyse", name, f"{p.kind}, {len(p.seq):,} letters")

    @reactive.effect
    def _log_batch():
        if run_batch.status() == "success" and not isinstance(run_batch.result(), str):
            r = run_batch.result()
            with reactive.isolate():
                job = batch_job()
            log("Batch", job[1] if job else "file", f"{r.n_ok:,} records analysed, {r.n_blocked:,} refused")

    @reactive.effect
    def _log_doc():
        if run_doc.status() == "success" and not isinstance(run_doc.result(), str):
            r = run_doc.result()
            with reactive.isolate():
                job = doc_job()
            log("Read a document", job[1] if job else "document",
                f"{len(r.found)} sequence{'s' if len(r.found) != 1 else ''} found by {r.source}")

    @render.ui
    def history_view():
        rows = history()
        if not rows:
            return notice("empty", "Nothing yet", "Analyses you run on any page appear here, with the time, the input "
                                                  "and the result.")
        return ui.output_data_frame("history_table")

    @render.data_frame
    def history_table():
        return render.DataGrid(pd.DataFrame(history()), height="520px", filters=True)

    @render.download(filename="codon_bench_session_history.csv")
    def dl_history():
        yield pd.DataFrame(history() or [log_row("", "", "")]).to_csv(index=False)

    @reactive.effect
    @reactive.event(input.clear_history)
    def _clear():
        history.set([])

    # ---- choosing what to analyse --------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.demo)
    def _demo():
        if input.demo() != "own":
            text = demos.SINGLE[input.demo()][1]
            ui.update_text_area("seq_text", value=text)
            ui.update_radio_buttons("strand", selected="template" if input.demo() == "template" else "coding")
            source.set(text)
            source_name.set("")

    @reactive.effect
    @reactive.event(input.go)
    def _go():
        source.set(input.seq_text())
        source_name.set("")

    @reactive.effect
    @reactive.event(input.seq_file)
    def _file():
        f = input.seq_file()
        if not f:
            return
        f = f[0]
        try:
            kind = F.sniff(f["datapath"], f["name"], f["size"])
            if kind.family != "sequence":
                raise F.Blocked("This box takes sequence files. Use the Batch page for tables, and 'Read a "
                                "document' for images and PDFs.")
            F.check_text_file(f["datapath"], kind)
            with F.open_text(f["datapath"], kind.gz) as fh:
                text = fh.read(SINGLE_LIMIT + 200_000)
            first = text
            if text.lstrip().startswith(">"):
                parts = text.lstrip().split("\n>", 1)
                first = parts[0]
            source.set(first)
            source_name.set(f["name"])
            ui.update_select("demo", selected="own")
            ui.update_text_area("seq_text", value=first if len(first) < 20000 else first[:20000] + "\n...")
        except F.Blocked as e:
            source.set("")
            source_name.set(f"__blocked__{e}")

    def send_to_analyzer(text: str):
        ui.update_select("demo", selected="own")
        ui.update_text_area("seq_text", value=text)
        source.set(text)
        source_name.set("")
        ui.update_navs("nav", selected="Analyse")

    @reactive.calc
    def parsed() -> S.Parsed | None:
        text = source()
        if source_name().startswith("__blocked__"):
            return S.Parsed("", "Unknown", reason=source_name()[11:])
        if not text or not text.strip():
            return None
        p = S.parse(text)
        if len(p.seq) > SINGLE_LIMIT:
            return S.Parsed("", "Unknown", reason=f"The sequence has {len(p.seq):,} letters. This page analyses up "
                                                  f"to {SINGLE_LIMIT:,}; use the Batch page for longer ones.")
        return p

    @reactive.calc
    def code() -> int:
        return int(input.code())

    @reactive.calc
    def coding_dna() -> str:
        """The coding strand, 5'->3', in DNA letters."""
        p = parsed()
        if not p or p.kind not in ("DNA", "RNA"):
            return ""
        if p.kind == "RNA":
            return S.to_dna(p.seq)
        return p.seq if input.strand() == "coding" else S.reverse_complement(p.seq)

    @reactive.calc
    def mrna() -> str:
        return S.to_rna(coding_dna())

    @reactive.calc
    def orfs():
        if not coding_dna():
            return []
        return S.find_orfs(coding_dna(), table_id=code(), min_aa=int(input.min_orf() or 30),
                           alt_starts=input.alt_starts())

    @reactive.calc
    def stats():
        p = parsed()
        if not p or p.kind not in ("DNA", "RNA"):
            return None
        return S.nucleotide_stats(p.seq, p.kind)

    @reactive.calc
    def protein() -> tuple[str, str]:
        """(protein, where it came from)."""
        p = parsed()
        if not p:
            return "", ""
        if p.kind == "Protein":
            return p.seq.rstrip("*"), "the protein you entered"
        o = orfs()
        if o:
            return o[0].protein, f"the longest ORF (frame {o[0].frame}, {o[0].start:,}–{o[0].end:,})"
        t = S.translate(mrna(), 0, code(), to_stop=True)
        return t, "a translation of frame +1 up to the first stop codon"

    # ---- header -----------------------------------------------------------------------------------
    @render.ui
    def analysis_head():
        p = parsed()
        if p is None:
            return notice("empty", "Paste a sequence to begin",
                          "Pick an example, paste DNA, RNA or protein letters, or upload a FASTA file, then press "
                          "Analyse sequence.")
        if p.kind == "Unknown":
            return notice("blocked", "This input was not analysed", p.reason + " Codon Bench only analyses DNA, RNA "
                          "and protein sequences.")
        name = p.header or source_name() or "Your sequence"
        items = [ui.span(name[:90], class_="name"), ui.span(p.kind, class_=f"kind kind-{p.kind}")]
        st = stats()

        def stat(v, label):
            return ui.span(ui.tags.b(v), ui.span(label), class_="stat")

        if st:
            items += [stat(f"{st['length']:,}", "bases"), stat(f"{st['gc_percent']:.1f}%", "GC content"),
                      stat(f"{len(orfs())}", f"ORFs of {int(input.min_orf() or 30)}+ aa")]
            if "tm" in st:
                items.append(stat(f"{st['tm']:.1f} °C", "melting temperature"))
        else:
            ps = S.protein_stats(p.seq)
            items += [stat(f"{len(p.seq):,}", "residues")]
            if ps:
                items += [stat(f"{ps.mw / 1000:,.2f} kDa", "molecular weight"), stat(f"{ps.pi:.2f}", "isoelectric point")]
        warn = []
        if p.ambiguous:
            warn.append(notice("warn", f"{p.ambiguous:,} ambiguous letters",
                               "Letters such as N, R or Y (or X in proteins) are kept but cannot be read as one base "
                               "or residue. Codons that contain them translate to X."))
        return ui.TagList(ui.div(*items, class_="identity"), *warn)

    # ---- body ------------------------------------------------------------------------------------
    @render.ui
    def analysis_body():
        p = parsed()
        if p is None or p.kind == "Unknown":
            return None
        n = len(p.seq)
        ribbon = ui.card(
            ui.card_header("Strand view" if p.kind != "Protein" else "Residue view"),
            ui.input_slider("rib_start", "Start position", 1, max(1, n - 39), 1, step=1, width="100%",
                            ticks=False) if n > 40 else ui.TagList(),
            ui.output_ui("ribbon"),
            ui.HTML(_ribbon_legend(p.kind)),
        )
        tabs = []
        if p.kind in ("DNA", "RNA"):
            tabs += [
                ui.nav_panel("Central dogma", ui.output_ui("dogma")),
                ui.nav_panel("Composition",
                             ui.layout_columns(
                                 ui.div(ui.h5("Base composition"), ui.output_ui("comp_plot")),
                                 ui.div(ui.h5("Properties"), ui.output_ui("comp_facts")), col_widths=(7, 5)),
                             ui.h5("GC content along the sequence"),
                             ui.p("Each point is the GC content of a window centred there. The dotted line is the "
                                  "whole-sequence average.", class_="small-note"),
                             ui.output_ui("gc_plot")),
                ui.nav_panel("Codons", ui.output_ui("codons_tab")),
                ui.nav_panel("Open reading frames", ui.output_ui("orf_view")),
                ui.nav_panel("ML predictions", ui.output_ui("ml_view")),
            ]
        tabs.append(ui.nav_panel("Protein", ui.output_ui("protein_view")))
        if p.kind == "Protein":
            tabs.insert(0, ui.nav_panel("Central dogma", ui.output_ui("dogma_protein")))
        return ui.TagList(ribbon, ui.navset_card_underline(*tabs, id="tabs"))

    @render.ui
    def ribbon():
        p = parsed()
        if not p or p.kind == "Unknown":
            return None
        start = int(input.rib_start()) if len(p.seq) > 40 and input.rib_start() else 1
        return ui.HTML(V.ribbon(p.seq, p.kind, start, 40))

    # ---- central dogma ---------------------------------------------------------------------------
    @render.ui
    def dogma():
        p = parsed()
        if not p or p.kind not in ("DNA", "RNA"):
            return None
        return ui.TagList(
            ui.layout_columns(
                ui.input_radio_buttons("read_from", "Read the mRNA from",
                                       {"aug": "the first AUG", "f0": "base 1", "f1": "base 2", "f2": "base 3"},
                                       selected="aug", inline=True),
                ui.output_ui("codon_slider"),
                col_widths=(5, 7)),
            ui.output_ui("dogma_table"),
            ui.output_ui("dogma_steps"),
        )

    @reactive.calc
    def codon_list():
        rf = input.read_from() if "read_from" in input else "aug"
        if rf == "aug":
            return S.translate_codons(mrna(), 0, code(), from_start=True, to_stop=True)
        return S.translate_codons(mrna(), int(rf[1]), code(), to_stop=True)

    @render.ui
    def codon_slider():
        c = codon_list()
        if not c:
            return ui.p("No AUG start codon in this sequence. Choose base 1, 2 or 3 instead.", class_="muted")
        return ui.input_slider("codon_focus", "Codon", 1, len(c), 1, step=1, width="100%",
                               animate=ui.AnimationOptions(interval=900, loop=False))

    @render.ui
    def dogma_table():
        c = codon_list()
        if not c:
            return None
        k = int(input.codon_focus()) if "codon_focus" in input and input.codon_focus() else 1
        return ui.HTML(V.dogma_alignment(coding_dna(), c, k))

    @render.ui
    def dogma_steps():
        p = parsed()
        dna = coding_dna()
        rna = mrna()
        prot = "".join(c.aa for c in codon_list()).rstrip("*")
        rt = S.reverse_transcribe(rna)
        rep = S.replicate(dna)
        lim = 3000

        def pre(s):
            return ui.tags.pre(V.wrap_seq(s[:lim], width=30) + ("\n..." if len(s) > lim else ""), class_="seq")

        def duplex(top, bottom, n=30):
            return ui.tags.pre(f"old 5' {top[:n]} 3'\nnew 3' {bottom[:n]} 5'" + ("\n(first 30 bases)" if len(top) > n else ""),
                               class_="seq")

        steps = [
            ui.div(ui.h4("1. Replication: DNA → DNA"),
                   ui.p("Each strand is a template, so each daughter helix keeps one old strand and one new one "
                        "(semi-conservative). DNA polymerase adds nucleotides only in the 5'→3' direction."),
                   ui.span("Daughter helix 1: the old coding strand and its new partner", class_="label-55"),
                   duplex(rep["parent_top"], rep["new_on_top"]),
                   ui.span("Daughter helix 2: the old template strand and its new partner", class_="label-55"),
                   ui.tags.pre(f"old 3' {rep['parent_bottom_3to5'][:30]} 5'\nnew 5' {rep['new_on_bottom'][:30]} 3'",
                               class_="seq"),
                   class_="step card p-3"),
            ui.div(ui.h4("2. Transcription: DNA → mRNA"),
                   ui.p("RNA polymerase reads the template strand 3'→5' and builds mRNA 5'→3'. The mRNA matches the "
                        "coding strand, with U in place of T." + (" You entered the template strand, so it was "
                                                                  "reverse-complemented first."
                                                                  if p.kind == "DNA" and input.strand() == "template"
                                                                  else "")),
                   ui.span("mRNA 5'→3'", class_="label-55"), pre(rna), class_="step card p-3"),
            ui.div(ui.h4("3. Translation: mRNA → protein"),
                   ui.p("The ribosome reads codons 5'→3'. Each tRNA pairs its anticodon with a codon and adds its amino "
                        "acid, until a stop codon."),
                   ui.span("Protein, N- to C-terminus", class_="label-55"), pre(prot or "(no protein in this frame)"),
                   class_="step card p-3"),
            ui.div(ui.h4("Reverse transcription: RNA → cDNA"),
                   ui.p("Reverse transcriptase (from retroviruses such as HIV, and used in RT-PCR) copies RNA into "
                        "complementary DNA. The first strand is complementary to the RNA; the second strand matches the "
                        "RNA with T in place of U."),
                   ui.span("cDNA first strand 5'→3'", class_="label-55"), pre(rt["cdna_first_strand"]),
                   ui.span("cDNA second strand 5'→3'", class_="label-55"), pre(rt["cdna_second_strand"]),
                   class_="step card p-3"),
        ]
        return ui.TagList(ui.div(*steps, class_="flow"),
                          ui.download_button("dl_products", "Download all products (FASTA)", class_="btn-ghost mt-3 w-auto"))

    @render.download(filename="codon_bench_products.fasta")
    def dl_products():
        dna, rna = coding_dna(), mrna()
        rt = S.reverse_transcribe(rna)
        prot = "".join(c.aa for c in codon_list()).rstrip("*")
        out = [("coding_strand_5to3", dna), ("template_strand_5to3", S.reverse_complement(dna)), ("mRNA_5to3", rna),
               ("cDNA_first_strand_5to3", rt["cdna_first_strand"]), ("protein", prot)]
        yield "".join(f">{n}\n{_plain_wrap(s)}\n" for n, s in out)

    @render.ui
    def dogma_protein():
        p = parsed()
        if not p:
            return None
        count, log10 = S.back_translation_count(p.seq, code())
        many = f"{count:,}" if 0 <= log10 < 15 else f"about 10^{log10:.0f}"
        first = p.seq[:12]
        rows = "".join(f"<tr><td>{V.aa_chip(a, i + 1)}</td><td>{', '.join(S.codons_for(a, code())) or '-'}</td></tr>"
                       for i, a in enumerate(first) if a in S.AMINO_20)
        return ui.TagList(
            notice("warn", "Translation cannot be reversed exactly",
                   f"Most amino acids have more than one codon, so this {len(p.seq):,}-residue protein could come from "
                   f"{many} different mRNA sequences. Paste DNA or RNA to see the full central dogma."),
            ui.h5("Possible codons for the first residues"),
            ui.HTML(f'<table class="table table-sm" style="max-width:520px"><thead><tr><th>Residue</th>'
                    f'<th>mRNA codons</th></tr></thead><tbody>{rows}</tbody></table>'))

    # ---- composition ---------------------------------------------------------------------------
    @render.ui
    @_plot
    def comp_plot():
        st = stats()
        req(st)
        return V.composition_bar(st["counts"], parsed().kind)

    @render.ui
    @_plot
    def gc_plot():
        p = parsed()
        st = stats()
        req(st)
        xs, ys, w = S.sliding_gc(p.seq)
        return V.gc_window(xs, ys, w, st["gc_percent"])

    @render.ui
    def comp_facts():
        st = stats()
        if not st:
            return None
        p = parsed()
        tu = "U" if p.kind == "RNA" else "T"
        items = [
            ("Length", f"{st['length']:,} bases"),
            ("GC content", f"{st['gc_percent']:.2f}%"),
            (f"A{tu} content", f"{st['at_percent']:.2f}%"),
            ("Purines (A + G)", f"{st['purines']:,}"),
            (f"Pyrimidines (C + {tu})", f"{st['pyrimidines']:,}"),
            ("GC skew (G − C)/(G + C)", f"{st['gc_skew']:+.3f}"),
            ("CpG dinucleotides", f"{st['cpg']:,}"),
            ("Molecular weight, single strand", f"{st['mw_single'] / 1000:,.1f} kDa"),
            ("Ambiguous letters", f"{st['ambiguous']:,}"),
        ]
        if "tm" in st:
            items += [("Melting temperature", f"{st['tm']:.1f} °C"), ("Method", st["tm_method"])]
        note = ui.p("In double-stranded DNA, A pairs with T (2 hydrogen bonds) and G with C (3), so GC-rich DNA melts "
                    "at a higher temperature. Chargaff's rule (A = T, G = C) holds for the double helix, not for one "
                    "strand.", class_="small-note mt-2")
        return ui.TagList(facts(items), note)

    # ---- codons ----------------------------------------------------------------------------------
    @render.ui
    def codons_tab():
        return ui.TagList(
            ui.input_radio_buttons("usage_frame", "Count codons in", {"0": "frame +1", "1": "frame +2", "2": "frame +3"},
                                   inline=True),
            ui.layout_columns(ui.output_ui("usage_grid"), ui.div(ui.h5("Codon by codon"),
                                                                 ui.output_data_frame("codon_table")),
                              col_widths=(6, 6)))

    @render.ui
    def usage_grid():
        f = int(input.usage_frame() or 0) if "usage_frame" in input else 0
        return ui.HTML(V.codon_usage_grid(dict(S.codon_usage(mrna(), f)), code()))

    @render.data_frame
    def codon_table():
        f = int(input.usage_frame() or 0) if "usage_frame" in input else 0
        cs = S.translate_codons(mrna(), f, code(), to_stop=False, limit=5000)
        df = pd.DataFrame([{"#": c.index, "Position": c.start, "Codon": c.codon,
                            "Anticodon 3'→5'": c.anticodon_3to5, "Amino acid": c.aa3, "Name": c.name,
                            "Class": c.aa_class} for c in cs])
        return render.DataGrid(df, height="420px", filters=False)

    # ---- ORFs ------------------------------------------------------------------------------------
    @render.ui
    def orf_view():
        o = orfs()
        if not o:
            return notice("empty", "No open reading frames of this length",
                          f"No stretch from a start codon to a stop codon codes for at least {input.min_orf()} amino "
                          "acids in any of the six frames. Lower the shortest ORF length in the sidebar.")
        return ui.TagList(
            ui.p(f"{len(o)} ORFs in six frames (three on this strand, three on the reverse complement). Select a row "
                 "to see its protein.", class_="muted"),
            ui.output_ui("orf_plot"),
            ui.output_data_frame("orf_table"),
            ui.output_ui("orf_detail"),
            ui.download_button("dl_orfs", "Download ORF proteins (FASTA)", class_="btn-ghost mt-2 w-auto"))

    @reactive.calc
    def orf_selected() -> int:
        try:
            rows = orf_table.cell_selection()["rows"]
            return int(rows[0]) if rows else 0
        except Exception:  # noqa: BLE001
            return 0

    @render.ui
    @_plot
    def orf_plot():
        return V.orf_map(orfs(), len(coding_dna()), orf_selected())

    @reactive.calc
    def orf_coding_probs() -> list[float]:
        if not ml.available():
            return []
        dna = coding_dna()
        out = []
        for o in orfs()[:200]:
            nuc = dna[o.start - 1:o.end]
            if o.frame.startswith("-"):
                nuc = S.reverse_complement(nuc)
            out.append(ml.coding_probability(nuc))
        return out

    @render.data_frame
    def orf_table():
        probs = orf_coding_probs()
        df = pd.DataFrame([{"ORF": i + 1, "Frame": o.frame, "Start": o.start, "End": o.end, "Length (nt)": o.length_nt,
                            "Protein (aa)": o.length_aa,
                            "Coding probability (ML)": (round(probs[i], 3) if i < len(probs) and probs[i] == probs[i]
                                                        else None),
                            "Begins": o.protein[:12] + ("…" if len(o.protein) > 12 else "")}
                           for i, o in enumerate(orfs()[:500])])
        return render.DataGrid(df, selection_mode="row", height="300px")

    @render.ui
    def orf_detail():
        o = orfs()
        if not o:
            return None
        k = orf_selected()
        x = o[min(k, len(o) - 1)]
        return ui.div(ui.h5(f"ORF {k + 1}: frame {x.frame}, {x.length_aa} amino acids"),
                      ui.tags.pre(_plain_wrap(x.protein), class_="seq"), class_="mt-2")

    @render.download(filename="codon_bench_orfs.faa")
    def dl_orfs():
        yield "".join(f">ORF{i + 1} frame={o.frame} {o.start}-{o.end} {o.length_aa}aa\n{_plain_wrap(o.protein)}\n"
                      for i, o in enumerate(orfs()))

    # ---- ML predictions --------------------------------------------------------------------------
    ML_SCAN_LIMIT = 30_000

    @reactive.calc
    def ml_scan():
        dna = coding_dna()
        if not dna or not ml.available():
            return None
        seg = dna[:ML_SCAN_LIMIT]
        d_pos, d_sc = ml.score_sites(seg, "donor")
        a_pos, a_sc = ml.score_sites(seg, "acceptor")
        tx, ty = ml.coding_track(seg)
        return {"seg": seg, "donor": (d_pos, d_sc), "acceptor": (a_pos, a_sc), "track": (tx, ty),
                "coding": ml.coding_probability(seg)}

    @render.ui
    def ml_view():
        if not ml.available():
            return notice("blocked", "Models not installed", "The models folder is missing. Run training/train_models.py.")
        r = ml_scan()
        if r is None:
            return None
        n = len(r["seg"])
        thr_c, thr_d, thr_a = ml.threshold("coding_potential"), ml.threshold("splice_donor"), ml.threshold("splice_acceptor")
        d_hits = [(p, s) for p, s in zip(*r["donor"]) if s >= thr_d]
        a_hits = [(p, s) for p, s in zip(*r["acceptor"]) if s >= thr_a]
        pc = r["coding"]
        tys = list(r["track"][1])
        frac = (sum(1 for y in tys if y >= thr_c) / len(tys)) if tys else float("nan")
        if pc != pc:
            verdict = "The sequence is too short to score coding potential (60 bases are needed)."
        elif tys and 0.1 <= frac <= 0.9:
            verdict = (f"{frac:.0%} of 150-base windows look protein-coding, so the sequence mixes coding stretches "
                       "(exons) with non-coding ones (introns, UTRs or intergenic DNA).")
        else:
            verdict = ("The sequence looks protein-coding." if pc >= thr_c else
                       "The sequence does not look protein-coding.")
        tiles = [("mean coding probability", "n/a" if pc != pc else f"{pc:.2f}"),
                 ("windows above the coding threshold", "n/a" if frac != frac else f"{frac:.0%}"),
                 ("predicted donor sites", f"{len(d_hits):,}"), ("predicted acceptor sites", f"{len(a_hits):,}"),
                 ("GT / AG candidates scored", f"{len(r['donor'][0]):,} / {len(r['acceptor'][0]):,}")]
        rows = sorted([("Donor (5' splice site)", p + 1, s, r["seg"][max(0, p - 3):p] + "|" + r["seg"][p:p + 6])
                       for p, s in d_hits] +
                      [("Acceptor (3' splice site)", p + 1, s, r["seg"][max(0, p - 10):p + 2] + "|" + r["seg"][p + 2:p + 5])
                       for p, s in a_hits], key=lambda x: x[1])
        df = pd.DataFrame([{"Site": k, "Position": pos, "Probability": round(float(sc), 3), "Sequence at the boundary": w}
                           for k, pos, sc, w in rows[:500]])
        parts = [
            ui.p(f"{verdict} Scores come from three machine-learning models trained on mouse and "
                 "tested on human genomes; see Model performance for how accurate they are. The strand you entered is "
                 "scanned as written" + (f" (first {ML_SCAN_LIMIT:,} bases)." if len(coding_dna()) > ML_SCAN_LIMIT else "."),
                 class_="muted"),
            ui.HTML('<div class="metric-row">' + "".join(f'<div class="metric"><b>{v}</b><span>{k}</span></div>'
                                                         for k, v in tiles) + "</div>"),
        ]
        if r["track"][0]:
            parts += [ui.h5("Coding potential along the sequence"), ui.output_ui("ml_coding_plot"),
                      ui.p(f"Each point scores a 150-base window. The dotted line is the decision threshold "
                           f"({thr_c:.2f}). Exons of a gene rise above it; introns and intergenic DNA stay below.",
                           class_="small-note")]
        parts += [ui.h5("Predicted splice sites"), ui.output_ui("ml_sites_plot"),
                  ui.p(f"Donor threshold {thr_d:.2f}, acceptor threshold {thr_a:.2f}. These sites need a GT or AG and "
                       "at least 40 bases on each side. In a cDNA or mRNA, real splice sites are already gone, so hits "
                       "there are cryptic sites that a mutation could activate.", class_="small-note")]
        if len(df):
            parts += [ui.output_data_frame("ml_sites_table"),
                      ui.download_button("dl_sites", "Download predicted sites (CSV)", class_="btn-ghost mt-2 w-auto")]
        else:
            parts.append(ui.p("No candidate scored above the thresholds.", class_="muted"))
        return ui.TagList(*parts)

    @render.ui
    @_plot
    def ml_coding_plot():
        r = ml_scan()
        req(r and r["track"][0])
        return V.coding_track_plot(r["track"][0], r["track"][1], ml.threshold("coding_potential"), len(r["seg"]))

    @render.ui
    @_plot
    def ml_sites_plot():
        r = ml_scan()
        req(r)
        return V.site_scan_plot(r["donor"], r["acceptor"], ml.threshold("splice_donor"),
                                ml.threshold("splice_acceptor"), len(r["seg"]))

    def _sites_df():
        r = ml_scan()
        thr_d, thr_a = ml.threshold("splice_donor"), ml.threshold("splice_acceptor")
        rows = [("Donor", p + 1, float(s)) for p, s in zip(*r["donor"]) if s >= thr_d] + \
               [("Acceptor", p + 1, float(s)) for p, s in zip(*r["acceptor"]) if s >= thr_a]
        return pd.DataFrame(sorted(rows, key=lambda x: x[1]), columns=["Site", "Position", "Probability"]).round(4)

    @render.data_frame
    def ml_sites_table():
        return render.DataGrid(_sites_df(), height="300px", filters=True)

    @render.download(filename="codon_bench_predicted_splice_sites.csv")
    def dl_sites():
        yield _sites_df().to_csv(index=False)

    # ---- protein ---------------------------------------------------------------------------------
    @render.ui
    def protein_view():
        prot, origin = protein()
        if not prot:
            return notice("empty", "No protein to describe", "No start codon was found, so nothing was translated.")
        ps = S.protein_stats(prot)
        if not ps:
            return notice("empty", "The protein is too short", "It needs at least two standard amino acids.")
        items = [
            ("Length", f"{ps.length:,} amino acids"),
            ("Molecular weight", f"{ps.mw:,.1f} Da ({ps.mw / 1000:,.2f} kDa)"),
            ("Isoelectric point (pI)", f"{ps.pi:.2f}"),
            ("Net charge at pH 7", f"{ps.charge_ph7:+.1f}"),
            ("Hydropathy (GRAVY)", f"{ps.gravy:+.3f} ({'hydrophobic' if ps.gravy > 0 else 'hydrophilic'} overall)"),
            ("Instability index", f"{ps.instability:.1f} ({'probably unstable' if ps.instability > 40 else 'probably stable'} "
                                  "in a test tube)"),
            ("Aromaticity", f"{ps.aromaticity:.3f}"),
            ("Extinction coefficient, 280 nm", f"{ps.extinction_reduced:,} (reduced cysteines), "
                                               f"{ps.extinction_cystines:,} (cystines) M⁻¹cm⁻¹"),
            ("Helix, turn, sheet (by residue type)", f"{ps.helix:.0%}, {ps.turn:.0%}, {ps.sheet:.0%}"),
        ]
        xs, ys = S.hydropathy(prot, 9)
        hyd = ui.output_ui("hyd_plot") if xs else ui.p("Too short for a hydropathy plot.", class_="muted")
        chips = "".join(V.aa_chip(a, i + 1) for i, a in enumerate(prot[:600]))
        return ui.TagList(
            ui.p(f"Describing {origin}.", class_="muted"),
            ui.layout_columns(ui.div(ui.h5("Properties"), facts(items),
                                     ui.p("Calculated with Biopython ProtParam (ExPASy methods). The helix, turn and "
                                          "sheet shares count residues that tend to form each structure; they are "
                                          "not a structure prediction.", class_="small-note mt-2")),
                              ui.div(ui.h5("Amino acid composition"), ui.HTML(_class_legend()),
                                     ui.output_ui("aa_plot")), col_widths=(5, 7)),
            ui.h5("Hydropathy"), hyd,
            ui.h5("Sequence"), ui.HTML(f'<div class="strand" style="flex-wrap:wrap;row-gap:4px">{chips}</div>'),
            ui.download_button("dl_protein", "Download protein (FASTA)", class_="btn-ghost mt-3 w-auto"))

    @render.ui
    @_plot
    def aa_plot():
        prot, _ = protein()
        ps = S.protein_stats(prot) if prot else None
        req(ps)
        return V.aa_composition(ps.composition)

    @render.ui
    @_plot
    def hyd_plot():
        prot, _ = protein()
        xs, ys = S.hydropathy(prot, 9)
        req(xs)
        return V.hydropathy_plot(xs, ys, 9)

    @render.download(filename="codon_bench_protein.faa")
    def dl_protein():
        prot, origin = protein()
        yield f">protein from {origin}\n{_plain_wrap(prot)}\n"

    # ---- genetic code page -----------------------------------------------------------------------
    @render.ui
    def aa_legend():
        return ui.HTML(_class_legend())

    @render.ui
    @_plot
    def wheel():
        q = (input.wheel_find() or "").strip()
        hl = _highlight(q, int(input.wheel_code()))
        return V.codon_wheel(int(input.wheel_code()), hl)

    # ---- batch ---------------------------------------------------------------------------------------
    batch_job = reactive.value(None)    # (path, name, size)

    @reactive.effect
    @reactive.event(input.batch_file)
    def _bf():
        f = input.batch_file()
        if f:
            batch_job.set((f[0]["datapath"], f[0]["name"], f[0]["size"]))

    for btn, fname in (("b_demo_fa", "sequences/mixed_sequences.fasta"), ("b_demo_csv", "sequences/mixed_sequences.csv"),
                       ("b_demo_xlsx", "sequences/mixed_sequences.xlsx"), ("b_demo_bad", "invalid/not_sequences.csv")):
        def _make(btn=btn, fname=fname):
            @reactive.effect
            @reactive.event(input[btn])
            def _():
                p = SAMPLES / fname
                batch_job.set((str(p), p.name, p.stat().st_size))
        _make()

    @reactive.extended_task
    async def run_batch(path, name, size, code_id, min_aa, key):
        def cb(frac, msg):
            PROGRESS[key] = (frac, msg)
        return await asyncio.to_thread(_safe_batch, path, name, size, code_id, min_aa, cb)

    batch_key = reactive.value("")

    @reactive.effect
    @reactive.event(batch_job, input.b_code, input.b_min_orf)
    def _start_batch():
        job = batch_job()
        if not job:
            return
        key = uuid.uuid4().hex
        batch_key.set(key)
        PROGRESS[key] = (0.0, "Starting")
        run_batch(*job, int(input.b_code()), int(input.b_min_orf() or 30), key)

    @render.ui
    def batch_status():
        st = run_batch.status()
        if st == "initial":
            return notice("empty", "Upload a file or try an example",
                          "Every record is checked first. Records that are not DNA, RNA or protein are listed but not "
                          "analysed, and a file with none is refused.")
        if st == "running":
            reactive.invalidate_later(0.5)
            frac, msg = PROGRESS.get(batch_key(), (0.0, "Working"))
            return ui.div(ui.h4("Reading the file"), ui.p(msg, class_="muted"),
                          ui.HTML(f'<div class="progress-line"><span style="width:{frac * 100:.0f}%"></span></div>'),
                          class_="notice notice-empty")
        if st == "error":
            return notice("blocked", "The batch could not be run", "Something went wrong while reading the file.")
        res = run_batch.result()
        if isinstance(res, str):
            return notice("blocked", "This file was refused", res)
        return None

    @render.ui
    def batch_body():
        if run_batch.status() != "success":
            return None
        r = run_batch.result()
        if isinstance(r, str):
            return None
        metrics = ui.HTML(
            '<div class="metric-row">' +
            "".join(f'<div class="metric"><b>{v}</b><span>{k}</span></div>' for k, v in [
                ("records analysed", f"{r.n_ok:,}"), ("records refused", f"{r.n_blocked:,}"),
                ("letters read", f"{r.total_letters:,}"),
                ("seconds", f"{r.seconds:.1f}" if r.seconds >= 0.1 else "under 0.1")]) + "</div>")
        kinds = ", ".join(f"{v:,} {k}" for k, v in r.kinds.items())
        notes = [ui.p(n, class_="small-note") for n in r.notes]
        charts = ui.layout_columns(
            ui.card(ui.card_header("Sequence lengths"), ui.output_ui("b_len")),
            ui.card(ui.card_header("GC content of DNA and RNA records"), ui.output_ui("b_gc")) if r.gcs
            else ui.card(ui.card_header("GC content"), ui.p("No DNA or RNA records.", class_="muted")),
            col_widths=(6, 6))
        downloads = ui.div(ui.download_button("dl_batch", "Download results (CSV)", class_="btn-primary"),
                           ui.download_button("dl_batch_prot", "Download proteins (FASTA)", class_="btn-ghost ms-2")
                           if r.proteins_path else ui.TagList(), class_="my-2")
        return ui.TagList(metrics, ui.p(f"Types: {kinds}.", class_="muted"), *notes, charts,
                          ui.card(ui.card_header("Records"), ui.output_data_frame("b_table")), downloads)

    @render.ui
    @_plot
    def b_len():
        r = run_batch.result()
        req(not isinstance(r, str) and r.lengths)
        spread = max(r.lengths) / max(1, min(r.lengths))
        return V.histogram(r.lengths, "Length (letters)", "#2a78d6", log=spread > 100)

    @render.ui
    @_plot
    def b_gc():
        r = run_batch.result()
        req(not isinstance(r, str) and r.gcs)
        return V.histogram(r.gcs, "GC content (%)", "#1baf7a")

    @render.data_frame
    def b_table():
        r = run_batch.result()
        df = pd.DataFrame(r.rows)
        df["description"] = [d if len(d) <= 40 else d[:39] + "…" for d in df["description"].astype(str)]
        df.columns = ["ID", "Description", "Type", "Length", "GC %", "Ambiguous", "Longest ORF (aa)", "ORF frame",
                      "Protein MW (kDa)", "Protein pI", "Status"]
        return render.DataGrid(df, height="420px", filters=True)

    @render.download(filename="codon_bench_batch.csv")
    def dl_batch():
        return run_batch.result().csv_path

    @render.download(filename="codon_bench_batch_proteins.faa")
    def dl_batch_prot():
        return run_batch.result().proteins_path

    # ---- documents -------------------------------------------------------------------------------
    doc_job = reactive.value(None)

    @reactive.effect
    @reactive.event(input.doc_file)
    def _df():
        f = input.doc_file()
        if f:
            doc_job.set((f[0]["datapath"], f[0]["name"], f[0]["size"]))

    for btn, fname in (("d_demo_page", "documents/textbook_page_hbb.png"), ("d_demo_photo", "documents/phone_photo_hbb.jpg"),
                       ("d_demo_table", "documents/peptide_table.png"), ("d_demo_pdf", "documents/sequence_handout.pdf"),
                       ("d_demo_bad", "invalid/not_related_poster.png")):
        def _mk(btn=btn, fname=fname):
            @reactive.effect
            @reactive.event(input[btn])
            def _():
                p = SAMPLES / fname
                doc_job.set((str(p), p.name, p.stat().st_size))
        _mk()

    @reactive.extended_task
    async def run_doc(path, name, size):
        return await asyncio.to_thread(_safe_doc, path, name, size)

    @reactive.effect
    @reactive.event(doc_job)
    def _start_doc():
        if doc_job():
            run_doc(*doc_job())

    @render.ui
    def doc_status():
        st = run_doc.status()
        if st == "initial":
            return notice("empty", "Upload a photo, scan or PDF of a sequence",
                          "For example a FASTA printout, a numbered sequence from a textbook, or a table of peptides. "
                          "Images with no sequence in them are refused.")
        if st == "running":
            return ui.div(ui.h4("Reading the document"), ui.p("Finding text, then working out what each line is. "
                                                             "This takes a few seconds per page.", class_="muted"),
                          class_="notice notice-empty")
        if st == "error":
            return notice("blocked", "The document could not be read", "Something went wrong while reading it.")
        r = run_doc.result()
        if isinstance(r, str):
            return notice("blocked", "This document was refused", r)
        return None

    @render.ui
    def doc_body():
        if run_doc.status() != "success" or isinstance(run_doc.result(), str):
            return None
        r: docread.Reading = run_doc.result()
        roles = {k: sum(1 for ln in r.lines if ln.role == k) for k in ("header", "sequence", "table", "text")}
        legend = ui.HTML('<div class="legend">' + "".join(
            f'<span><span class="sw" style="background:rgb{docread.ROLE_COLORS[k]}"></span>{label} ({roles[k]})</span>'
            for k, label in [("header", "Header"), ("sequence", "Sequence line"), ("table", "Table row"),
                             ("text", "Other text")]) + "</div>")
        imgs = [ui.tags.img(src=_png_uri(im), class_="overlay", alt=f"Page {i + 1} with the lines it found outlined")
                for i, im in enumerate(r.overlays[:3])]
        found = []
        for i, f in enumerate(r.found):
            found.append(ui.div(
                ui.h4(f.name), ui.p(f"{f.kind}, {len(f.sequence):,} letters, from {f.lines} line"
                                    f"{'s' if f.lines != 1 else ''} on page {f.page}.", class_="muted mb-1"),
                ui.HTML(f'<div class="conf-bar" title="Reading confidence {f.confidence:.0%}"><span style="width:'
                        f'{f.confidence * 100:.0f}%"></span></div>'),
                ui.span(f"Reading confidence {f.confidence:.0%}" + (" (text taken directly from the PDF)"
                                                                     if r.source == "PDF text" else ""),
                        class_="small-note"),
                ui.input_text_area(f"found_{i}", "Check and correct before analysing", value=f">{f.name}\n" +
                                   _plain_wrap(f.sequence), rows=4, width="100%"),
                ui.input_action_button(f"use_{i}", "Analyse this sequence", class_="btn-primary"),
                class_="found"))
        return ui.layout_columns(
            ui.div(ui.h4(r.doc_type), ui.p(f"Read with {r.source}, {r.pages} page{'s' if r.pages != 1 else ''}. "
                                           f"{len(r.found)} sequence{'s' if len(r.found) != 1 else ''} found.",
                                           class_="muted"), *found),
            ui.div(legend, *imgs) if imgs else ui.div(legend, ui.p("No image to show: the PDF held text.",
                                                                   class_="muted")),
            col_widths=(5, 7))

    @reactive.effect
    def _use_found():
        if run_doc.status() != "success" or isinstance(run_doc.result(), str):
            return
        for i in range(len(run_doc.result().found)):
            _bind_use(i)

    bound: set[int] = set()

    def _bind_use(i):
        if i in bound:
            return
        bound.add(i)

        @reactive.effect
        @reactive.event(input[f"use_{i}"], ignore_init=True)
        def _():
            send_to_analyzer(input[f"found_{i}"]())

    mutation_server(input, output, session, send_to_analyzer, log)
    performance_server(input, output, session)


def _safe_batch(path, name, size, code_id, min_aa, cb):
    try:
        return B.run(path, name, size, code_id, min_aa, cb)
    except F.Blocked as e:
        return str(e)


def _safe_doc(path, name, size):
    try:
        kind = F.sniff(path, name, size)
        if kind.family == "image":
            return docread.read_image(path)
        if kind.family == "pdf":
            return docread.read_pdf(path)
        return "This page reads images and PDFs. Sequence and table files go on the Analyse or Batch page."
    except F.Blocked as e:
        return str(e)


def _plain_wrap(s: str, n: int = 60) -> str:
    return "\n".join(s[i:i + n] for i in range(0, len(s), n))


def _png_uri(im) -> str:
    buf = io.BytesIO()
    im2 = im.copy()
    im2.thumbnail((1400, 1400))
    im2.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ribbon_legend(kind: str) -> str:
    if kind == "Protein":
        return _class_legend()
    bases = ["A", "C", "G", "U" if kind == "RNA" else "T"]
    items = "".join(f'<span>{V.base_chip(b)}{S.BASES[b]["name"]}</span>' for b in bases)
    return (f'<div class="ribbon-legend">{items}<span>Bars between strands: hydrogen bonds (A–T and A–U have 2, '
            f'G–C has 3)</span></div>')


def _class_legend() -> str:
    names = {"Nonpolar": "Nonpolar", "Polar": "Polar, uncharged", "Acidic": "Acidic (negative)",
             "Basic": "Basic (positive)", "Stop": "Stop"}
    return '<div class="legend">' + "".join(
        f'<span><span class="sw" style="background:var(--{ {"Nonpolar": "G", "Polar": "T", "Acidic": "U", "Basic": "C"}.get(k, "stop")})"></span>{v}</span>'
        for k, v in names.items()
    ) + "</div>"


def _highlight(q: str, table_id: int) -> set[str] | None:
    if not q:
        return None
    q = q.strip()
    up = q.upper().replace("T", "U") if len(q) == 3 and set(q.upper()) <= set("ACGTU") else None
    if up:
        return {up}
    if q.lower() == "stop":
        return set(S.codons_for("*", table_id))
    for one, (three, name, _cls) in S.AMINO_ACIDS.items():
        if q.lower() in (three.lower(), name.lower()) or (len(q) == 1 and q.upper() == one):
            return set(S.codons_for(one, table_id))
    return set()


app = App(app_ui, server, static_assets=ROOT / "www")
