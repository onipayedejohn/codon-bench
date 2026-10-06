"""Train, validate and export the three models.

Protocol
  1. Mouse chr19 (GENCODE M31): 5-fold cross-validation grouped by gene, comparing a logistic
     regression baseline with LightGBM. Out-of-fold predictions choose the decision threshold
     (the one that maximises F1).
  2. LightGBM, which beat the baseline in every fold, is refitted on all mouse data and exported to ONNX.
  3. Human chr21 (GENCODE) is never used for training or tuning. It is the held-out test set:
     window-level metrics, a SpliceAI-style top-k accuracy over every GT/AG in each gene, and a
     transcript-level coding vs lncRNA test.

Run from the project folder:  python training/train_models.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
                             accuracy_score, brier_score_loss)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from codonbench import ml  # noqa: E402
from training import build_datasets as BD  # noqa: E402

PROC, MODELS = ROOT / "training" / "processed", ROOT / "models"
SEED = 20261006

LGB_PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                  random_state=SEED, n_jobs=2, verbose=-1)
FEATURES = {"splice_donor": ml.donor_features, "splice_acceptor": ml.acceptor_features,
            "coding_potential": ml.coding_features}


def featurise(windows, fx):
    return np.vstack([fx(w) for w in windows]).astype(np.float32)


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "positives": int(y.sum()), "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)), "accuracy": float(accuracy_score(y, yhat)),
            "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat)), "specificity": float(tn / max(1, tn + fp)),
            "f1": float(f1_score(y, yhat)), "mcc": float(matthews_corrcoef(y, yhat)),
            "brier": float(brier_score_loss(y, p)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def curves(y, p, n=120):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    k1 = np.linspace(0, len(fpr) - 1, min(n, len(fpr))).astype(int)
    k2 = np.linspace(0, len(prec) - 1, min(n, len(prec))).astype(int)
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="uniform")
    return {"roc": {"fpr": fpr[k1].round(4).tolist(), "tpr": tpr[k1].round(4).tolist()},
            "pr": {"recall": rec[k2].round(4).tolist(), "precision": prec[k2].round(4).tolist()},
            "calibration": {"predicted": mean.round(4).tolist(), "observed": frac.round(4).tolist()}}


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(1e-9, prec + rec)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def cross_validate(X, y, groups, name):
    gkf = GroupKFold(n_splits=5)
    oof = {"logistic_regression": np.zeros(len(y)), "lightgbm": np.zeros(len(y))}
    folds, iters = [], []
    for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        lr.fit(X[tr], y[tr])
        oof["logistic_regression"][va] = lr.predict_proba(X[va])[:, 1]
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        inner = np.random.default_rng(SEED + k).random(len(tr)) < 0.1
        m.fit(X[tr][~inner], y[tr][~inner], eval_set=[(X[tr][inner], y[tr][inner])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        iters.append(m.best_iteration_ or LGB_PARAMS["n_estimators"])
        oof["lightgbm"][va] = m.predict_proba(X[va], num_iteration=m.best_iteration_)[:, 1]
        folds.append({mdl: {"auroc": float(roc_auc_score(y[va], oof[mdl][va])),
                            "auprc": float(average_precision_score(y[va], oof[mdl][va]))} for mdl in oof})
        print(f"  {name} fold {k + 1}: " + ", ".join(f"{m_}: AUROC {v['auroc']:.4f} AUPRC {v['auprc']:.4f}"
                                                   for m_, v in folds[-1].items()))
    summary = {}
    for mdl in oof:
        a = [f[mdl]["auroc"] for f in folds]
        b = [f[mdl]["auprc"] for f in folds]
        summary[mdl] = {"auroc_mean": float(np.mean(a)), "auroc_sd": float(np.std(a)),
                        "auprc_mean": float(np.mean(b)), "auprc_sd": float(np.std(b))}
    return oof, summary, int(np.mean(iters))


def export_onnx(model, n_features, path):
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    onx = onnxmltools.convert_lightgbm(model, initial_types=[("features", FloatTensorType([None, n_features]))],
                                       zipmap=False, target_opset=15)
    path.write_bytes(onx.SerializeToString())


def topk_accuracy(kind, booster_predict):
    """SpliceAI-style: in each human gene, score every GT (or AG); of the k annotated sites, how many
    are among the k highest-scoring candidates?"""
    genome = BD.read_fasta(BD.RAW / "hs_chr21.fa")
    tx = BD.read_gtf(BD.RAW / "hs_chr21.gtf")
    donors, acceptors, gene_sites, spans, _ = BD.splice_sites(genome, tx)
    up, down = (ml.DONOR_UP, ml.DONOR_DOWN) if kind == "donor" else (ml.ACCEPTOR_UP, ml.ACCEPTOR_DOWN)
    up, down = up + ml.CONTEXT, down + ml.CONTEXT
    hit = total = n_cand = 0
    fx = FEATURES[f"splice_{kind}"]
    for key, sites in gene_sites.items():
        true = sites[kind]
        if not true:
            continue
        chrom, strand, _ = key
        g = genome[chrom]
        lo, hi = spans[key]
        cand = BD.candidates(g, lo, hi, strand, kind)
        wins, keep = [], []
        for c in cand:
            w = BD.oriented(g, int(c), strand, up, down)
            if len(w) == up + down and "N" not in w[ml.CONTEXT:-ml.CONTEXT]:
                wins.append(w)
                keep.append(int(c))
        if not wins:
            continue
        p = booster_predict(featurise(wins, fx))
        k = len(true)
        top = {keep[i] for i in np.argsort(-p)[:k]}
        hit += len(top & true)
        total += k
        n_cand += len(keep)
    return {"top_k_accuracy": hit / max(1, total), "annotated_sites": total, "candidates_scored": n_cand}


def importance(model, names, kind):
    gain = model.booster_.feature_importance(importance_type="gain")
    gain = gain / gain.sum()
    top = np.argsort(-gain)[:15]
    out = {"top_features": [{"feature": names[i], "share": float(gain[i])} for i in top]}
    if kind in ("donor", "acceptor"):
        up = ml.DONOR_UP if kind == "donor" else ml.ACCEPTOR_UP
        L = (ml.DONOR_UP + ml.DONOR_DOWN) if kind == "donor" else (ml.ACCEPTOR_UP + ml.ACCEPTOR_DOWN)
        pos = gain[:4 * L].reshape(L, 4).sum(axis=1)
        out["by_position"] = {"offset": [i - up for i in range(L)], "share": pos.round(5).tolist()}
    return out


def main():
    MODELS.mkdir(exist_ok=True)
    stats = json.loads((PROC / "dataset_stats.json").read_text())
    card = {"created": time.strftime("%Y-%m-%d"), "python": platform.python_version(),
            "lightgbm": lgb.__version__, "seed": SEED, "datasets": stats,
            "data_sources": {
                "train": "Mouse chromosome 19, GRCm39, GENCODE M31 (Ensembl 108) annotation",
                "test": "Human chromosome 21, GRCh38, GENCODE annotation (272 genes, 367 transcripts)",
                "provider": "nf-core/test-datasets (branch modules), github.com/nf-core/test-datasets"},
            "models": {}}
    curves_out = {}
    for name in ("splice_donor", "splice_acceptor", "coding_potential"):
        t0 = time.time()
        base = name.replace("splice_", "") if name.startswith("splice") else "coding"
        tr = pd.read_csv(PROC / (f"splice_{base}_mouse.csv.gz" if name.startswith("splice") else "coding_mouse.csv.gz"))
        te = pd.read_csv(PROC / (f"splice_{base}_human.csv.gz" if name.startswith("splice") else "coding_human.csv.gz"))
        fx = FEATURES[name]
        if name.startswith("splice"):
            # 10% of training windows lose one flank, so the model copes with sites near the ends of a sequence
            rng = np.random.default_rng(SEED)
            w = tr.window.to_numpy().copy()
            for i in np.where(rng.random(len(w)) < 0.10)[0]:
                cut = int(rng.integers(20, ml.CONTEXT + 1))
                w[i] = ("N" * cut + w[i][cut:]) if rng.random() < 0.5 else (w[i][:-cut] + "N" * cut)
            tr = tr.assign(window=w)
        Xtr, ytr = featurise(tr.window, fx), tr.label.to_numpy()
        Xte, yte = featurise(te.window, fx), te.label.to_numpy()
        print(f"{name}: train {Xtr.shape}, test {Xte.shape}")
        oof, cv, n_iter = cross_validate(Xtr, ytr, tr.gene_id.to_numpy(), name)
        thr = best_threshold(ytr, oof["lightgbm"])
        final = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": n_iter})
        final.fit(Xtr, ytr)
        p_lgb = final.predict_proba(Xte)[:, 1]
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(Xtr, ytr)
        p_lr = lr.predict_proba(Xte)[:, 1]
        thr_lr = best_threshold(ytr, oof["logistic_regression"])
        export_onnx(final, Xtr.shape[1], MODELS / f"{name}.onnx")
        ml._session.cache_clear()
        p_onnx = ml.predict(name, Xte)
        parity = float(np.max(np.abs(p_onnx - p_lgb)))
        entry = {
            "task": {"splice_donor": "Is this GT the start of an intron (5' splice site)?",
                     "splice_acceptor": "Is this AG the end of an intron (3' splice site)?",
                     "coding_potential": "Is this 150-base DNA window protein-coding?"}[name],
            "algorithm": "LightGBM gradient-boosted trees", "n_trees": n_iter,
            "n_features": int(Xtr.shape[1]), "threshold": thr,
            "threshold_rule": "maximises F1 on mouse out-of-fold predictions",
            "cross_validation_mouse": cv,
            "out_of_fold_mouse": metrics(ytr, oof["lightgbm"], thr),
            "test_human": metrics(yte, p_lgb, thr),
            "baseline_logistic_regression_test_human": metrics(yte, p_lr, thr_lr),
            "onnx_max_abs_difference": parity,
            "importance": importance(final, ml.FEATURE_NAMES[name.replace("splice_", "") if "splice" in name
                                                              else "coding"], base)}
        if name.startswith("splice"):
            entry["test_human_top_k"] = topk_accuracy(base, lambda X: final.predict_proba(X)[:, 1])
            pos = tr[tr.label == 1].window.str.slice(ml.CONTEXT, -ml.CONTEXT)
            L = len(pos.iloc[0])
            freq = np.zeros((L, 4))
            for w in pos:
                for i, ch in enumerate(w):
                    j = "ACGT".find(ch)
                    if j >= 0:
                        freq[i, j] += 1
            freq /= freq.sum(axis=1, keepdims=True)
            entry["consensus"] = {"offset": [i - (ml.DONOR_UP if base == "donor" else ml.ACCEPTOR_UP) for i in range(L)],
                                  "freq": freq.round(4).tolist()}
        card["models"][name] = entry
        curves_out[name] = {"test_human": curves(yte, p_lgb), "test_human_baseline": curves(yte, p_lr),
                            "oof_mouse": curves(ytr, oof["lightgbm"])}
        print(f"  {name} test human: AUROC {entry['test_human']['auroc']:.4f} AUPRC {entry['test_human']['auprc']:.4f}"
              f" F1 {entry['test_human']['f1']:.4f} MCC {entry['test_human']['mcc']:.4f}; ONNX diff {parity:.2e};"
              f" {time.time() - t0:.0f}s")
    # transcript-level coding test with the exported model (same code path as the app)
    ml.model_card.cache_clear()
    tt = pd.read_csv(PROC / "coding_transcripts_human.csv.gz")
    pt = np.array([ml.coding_probability(s) for s in tt.sequence])
    thr = card["models"]["coding_potential"]["threshold"]
    card["models"]["coding_potential"]["test_human_transcripts"] = metrics(tt.label.to_numpy(), pt, thr)
    curves_out["coding_potential"]["test_human_transcripts"] = curves(tt.label.to_numpy(), pt)
    print("  coding transcripts human:", {k: round(v, 4) for k, v in
                                          card["models"]["coding_potential"]["test_human_transcripts"].items()
                                          if isinstance(v, float)})
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=1))
    (MODELS / "curves.json").write_text(json.dumps(curves_out))


if __name__ == "__main__":
    main()
