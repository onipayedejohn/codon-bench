"""The Model performance page, built from models/model_card.json and models/curves.json."""
from __future__ import annotations

import json

from shiny import render, ui

from . import ml
from . import viz as V

_plot = V.as_plot
NAMES = {"splice_donor": "Splice donor (5' site)", "splice_acceptor": "Splice acceptor (3' site)",
         "coding_potential": "Coding potential"}


def _curves() -> dict:
    p = ml.MODEL_DIR / "curves.json"
    return json.loads(p.read_text()) if p.exists() else {}


def performance_ui():
    return ui.nav_panel(
        "Model performance",
        ui.h2("Model performance", class_="page-title"),
        ui.p("Three gradient-boosted tree models (LightGBM, run here through ONNX Runtime) power the ML features. Each "
             "was trained on mouse chromosome 19 and tested on human chromosome 21, which the models never saw during "
             "training or tuning, so the test numbers measure how well they generalise to a new species.",
             class_="lede"),
        ui.output_ui("perf_overview"),
        ui.navset_card_underline(
            *[ui.nav_panel(NAMES[k], ui.output_ui(f"perf_{k}")) for k in NAMES],
            ui.nav_panel("Data and method", ui.output_ui("perf_method")),
            id="perf_tabs"),
    )


def performance_server(input, output, session):
    card = ml.model_card()
    curves = _curves()

    @render.ui
    def perf_overview():
        if not card:
            return ui.div(ui.h4("Models not found"), ui.p("Run training/train_models.py to create them."),
                          class_="notice notice-blocked")
        tiles = []
        for k, label in NAMES.items():
            t = card["models"][k]["test_human"]
            tiles.append(f'<div class="metric"><b>{t["auroc"]:.3f}</b><span>{label}: AUROC on human</span></div>')
        t = card["models"]["coding_potential"].get("test_human_transcripts")
        if t:
            tiles.append(f'<div class="metric"><b>{t["accuracy"]:.1%}</b><span>Coding vs lncRNA transcripts, '
                         f'accuracy on human</span></div>')
        return ui.HTML('<div class="metric-row">' + "".join(tiles) + "</div>")

    def model_page(key: str):
        m = card["models"][key]
        t, b, oof = m["test_human"], m["baseline_logistic_regression_test_human"], m["out_of_fold_mouse"]
        cv = m["cross_validation_mouse"]

        def row(name, d, extra=""):
            return (f"<tr><th scope='row'>{name}</th><td>{d['auroc']:.3f}</td><td>{d['auprc']:.3f}</td>"
                    f"<td>{d['f1']:.3f}</td><td>{d['mcc']:.3f}</td><td>{d['precision']:.3f}</td>"
                    f"<td>{d['recall']:.3f}</td><td>{d['specificity']:.3f}</td><td>{extra}</td></tr>")
        rows = [row("LightGBM, human test", t, f"{t['n']:,} ({t['positives']:,} positive)"),
                row("Logistic regression baseline, human test", b, "same windows"),
                row("LightGBM, mouse out-of-fold", oof, f"{oof['n']:,} ({oof['positives']:,} positive)")]
        if "test_human_transcripts" in m:
            tt = m["test_human_transcripts"]
            rows.append(row("LightGBM, whole human transcripts", tt, f"{tt['n']} ({tt['positives']} coding)"))
        table = ("<div class='table-scroll'><table class='table table-sm metrics-table'><thead><tr><th>Evaluation</th>"
                 "<th>AUROC</th><th>AUPRC</th><th>F1</th><th>MCC</th><th>Precision</th><th>Recall</th>"
                 "<th>Specificity</th><th>Examples</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
        cvtxt = ", ".join(f"{name.replace('_', ' ')}: AUROC {v['auroc_mean']:.3f} ± {v['auroc_sd']:.3f}, AUPRC "
                          f"{v['auprc_mean']:.3f} ± {v['auprc_sd']:.3f}" for name, v in cv.items())
        cm = t["confusion"]
        confusion = (f"<table class='table table-sm confusion'><thead><tr><th></th><th>Predicted negative</th>"
                     f"<th>Predicted positive</th></tr></thead><tbody>"
                     f"<tr><th scope='row'>Actual negative</th><td>{cm['tn']:,}</td><td>{cm['fp']:,}</td></tr>"
                     f"<tr><th scope='row'>Actual positive</th><td>{cm['fn']:,}</td><td>{cm['tp']:,}</td></tr>"
                     f"</tbody></table>")
        prevalence = t["positives"] / t["n"]
        notes = [ui.p(m["task"], class_="lede mb-2"),
                 ui.p(f"Decision threshold {m['threshold']:.2f} ({m['threshold_rule']}). {m['n_trees']} trees, "
                      f"{m['n_features']} features. ONNX export matches LightGBM to within "
                      f"{m['onnx_max_abs_difference']:.1e}.", class_="small-note")]
        if "test_human_top_k" in m:
            k = m["test_human_top_k"]
            notes.append(ui.p(f"Top-k accuracy on human genes: {k['top_k_accuracy']:.1%}. In each gene, every "
                              f"{'GT' if 'donor' in key else 'AG'} was scored ({k['candidates_scored']:,} candidates in "
                              f"all) and the k highest were compared with the k annotated sites ({k['annotated_sites']:,}). "
                              "This is the stringent genome-scan measure used by SpliceAI; it is much harder than the "
                              "window test, where there are 25 decoys per real site.", class_="small-note"))
        parts = [*notes, ui.HTML(table), ui.p(f"5-fold cross-validation on mouse, grouped by gene: {cvtxt}.",
                                              class_="small-note"),
                 ui.layout_columns(
                     ui.card(ui.card_header("ROC curve"), ui.output_ui(f"roc_{key}")),
                     ui.card(ui.card_header("Precision-recall curve"), ui.output_ui(f"pr_{key}"),
                             ui.p(f"Dotted line: precision of random guessing ({prevalence:.1%} of test windows are "
                                  "positive).", class_="small-note px-3")),
                     col_widths=(6, 6)),
                 ui.layout_columns(
                     ui.card(ui.card_header("Calibration on human"), ui.output_ui(f"cal_{key}"),
                             ui.p("Points near the diagonal mean a score of 0.7 is right about 70% of the time.",
                                  class_="small-note px-3")),
                     ui.card(ui.card_header("Confusion matrix on human, at the threshold"), ui.HTML(confusion),
                             ui.p(f"Accuracy {t['accuracy']:.3f}. Brier score {t['brier']:.4f}.", class_="small-note px-3")),
                     col_widths=(6, 6))]
        if "by_position" in m["importance"]:
            parts.append(ui.card(ui.card_header("What the model learned"), ui.output_ui(f"pos_{key}"),
                                 ui.p("The top panel shows where the model's decisions draw on the sequence; the bottom "
                                      "panel shows how conserved each position is across real sites. The peaks line up "
                                      "with the known splice consensus (MAG|GURAGU for donors; the polypyrimidine tract "
                                      "and YAG|G for acceptors), which the model found without being told.",
                                      class_="small-note px-3")))
        parts.append(ui.card(ui.card_header("Most important features"), ui.output_ui(f"imp_{key}")))
        return ui.TagList(*parts)

    def register(oid: str, fn, plot: bool = False):
        fn.__name__ = oid
        render.ui(_plot(fn) if plot else fn)

    for key in NAMES:
        def _bind(key=key):
            register(f"perf_{key}", lambda: model_page(key) if card else None)
            base = {"test_human": "LightGBM, human", "test_human_baseline": "Logistic regression, human",
                    "oof_mouse": "LightGBM, mouse (cross-validated)"}
            roc_names = dict(base)
            if key == "coding_potential":
                roc_names["test_human_transcripts"] = "LightGBM, whole transcripts"
            t = card["models"][key]["test_human"] if card else {"positives": 1, "n": 2}
            register(f"roc_{key}", lambda: V.roc_plot(curves[key], roc_names), plot=True)
            register(f"pr_{key}", lambda: V.pr_plot(curves[key], base, t["positives"] / t["n"]), plot=True)
            register(f"cal_{key}", lambda: V.calibration_plot(curves[key]["test_human"]["calibration"]), plot=True)
            register(f"imp_{key}", lambda: V.importance_bar(card["models"][key]["importance"]["top_features"]),
                     plot=True)
            if key.startswith("splice"):
                register(f"pos_{key}", lambda: V.position_importance_plot(
                    card["models"][key]["importance"]["by_position"], card["models"][key]["consensus"],
                    key.replace("splice_", "")), plot=True)
        _bind()

    @render.ui
    def perf_method():
        if not card:
            return None
        ds = card["datasets"]
        mo, hu = ds["mouse"], ds["human"]
        src = card["data_sources"]
        return ui.markdown(f"""
**Data.** Training: {src['train']}. Test: {src['test']}. Both come from {src['provider']}, with checksums recorded in
`training/download_data.py`.

**Splice sites.** Every canonical GT-AG intron boundary of every annotated transcript is a positive
({mo['donor_pos']:,} donors and {mo['acceptor_pos']:,} acceptors in mouse; {hu['donor_pos']:,} and {hu['acceptor_pos']:,}
in human). Negatives are GT or AG dinucleotides inside the same genes that are not an annotated boundary of any
transcript, 25 per positive. Introns with non-canonical ends ({mo['non_canonical_introns_skipped']} in mouse) are left
out. Features: the bases in an 80-base (donor) or 100-base (acceptor) window, 3-mer composition on each side, the
polypyrimidine tract and branch-point region, and the composition of 160 bases of flanking sequence on each side.

**Coding potential.** Positives are 150-base windows from the spliced coding sequence of one transcript per
protein-coding gene, at random offsets so all three reading frames appear ({mo['coding_pos']:,} in mouse). Negatives,
in equal number, come from introns, intergenic DNA on either strand, UTRs and spliced lncRNA exons. Features: stop-codon
rates in the three frames, codon usage and base composition of the frame with the fewest stops, 1- to 3-mer frequencies,
the Fickett TESTCODE score and the period-3 Fourier signal.

**Method.** 5-fold cross-validation on mouse, grouped by gene so no gene appears in both training and validation, compared
a logistic regression baseline with LightGBM. LightGBM won on every model. The number of trees came from early stopping;
the decision threshold maximises F1 on the out-of-fold predictions. The final model was refitted on all mouse data and
then scored on human. The human set was scored twice in development: for a first version, and after adding the
flanking-context features, a change chosen because it raised mouse cross-validated AUPRC. Seed {card['seed']}. LightGBM {card['lightgbm']}, Python {card['python']}.

**Limits.** The splice models look at about 400 bases, so they miss long-range signals that deep models such as
SpliceAI (10,000 bases) use; their top-k accuracy is correspondingly lower. Scores are probabilities at a 1-in-26
prevalence of real sites among candidates, so in a whole-gene scan most high scores are still decoys. The human test set
is a subset of chromosome 21 ({hu['transcripts']} transcripts). None of the models is validated for clinical use.
""")
