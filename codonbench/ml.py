"""Machine-learning models: splice-site prediction and coding potential.

Feature extraction lives here so that training (training/train_models.py) and the app use exactly the
same code. Models are LightGBM classifiers exported to ONNX and run with onnxruntime.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Window geometry, in transcript orientation. Offset 0 is the first base of the GT (donor) or AG
# (acceptor) dinucleotide.
DONOR_UP, DONOR_DOWN = 40, 40          # 40 exon bases, then GT + 38 intron bases
ACCEPTOR_UP, ACCEPTOR_DOWN = 60, 40    # 60 intron bases, then AG + 38 exon bases
CONTEXT = 160                          # extra bases on each side, summarised as composition (exon-like or not)
CODING_WINDOW = 150

_LUT = np.full(256, 4, dtype=np.int8)
for _i, _b in enumerate("ACGT"):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def encode(s: str) -> np.ndarray:
    """A=0 C=1 G=2 T=3, anything else 4."""
    return _LUT[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]


def _kmer_index(codes: np.ndarray, k: int) -> np.ndarray:
    """Index of every k-mer (base 4); -1 where the k-mer contains a non-ACGT base."""
    n = len(codes) - k + 1
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.zeros(n, dtype=np.int64)
    bad = np.zeros(n, dtype=bool)
    for j in range(k):
        c = codes[j:j + n].astype(np.int64)
        bad |= c == 4
        idx = idx * 4 + np.where(c == 4, 0, c)
    idx[bad] = -1
    return idx


def kmer_freq(codes: np.ndarray, k: int) -> np.ndarray:
    idx = _kmer_index(codes, k)
    idx = idx[idx >= 0]
    out = np.bincount(idx, minlength=4 ** k).astype(np.float32)
    return out / max(1, len(idx))


# ---------------------------------------------------------------------------------------------
# Splice sites
# ---------------------------------------------------------------------------------------------
def _onehot(codes: np.ndarray) -> np.ndarray:
    eye = np.vstack([np.eye(4, dtype=np.float32), np.full((1, 4), 0.25, dtype=np.float32)])
    return eye[codes].ravel()


def _context_features(seg: str) -> np.ndarray:
    """Composition of a flanking stretch: does it look like exon or intron? N (missing) bases are ignored."""
    c = encode(seg)
    valid = c[c < 4]
    stops = []
    for f in range(3):
        cod = _codons(c, f)
        cod = cod[cod >= 0]
        stops.append(sum(int(x in _STOPS) for x in cod) / max(1, len(cod)))
    gc = np.mean(np.isin(valid, (1, 2))) if len(valid) else 0.0
    return np.concatenate([kmer_freq(c, 3), np.sort(np.array(stops, dtype=np.float32)),
                           np.array([gc, len(valid) / max(1, len(c))], dtype=np.float32)])


def donor_features(window: str) -> np.ndarray:
    """window: CONTEXT | exon(40) | GT + intron(38) | CONTEXT."""
    left, right = window[:CONTEXT], window[-CONTEXT:]
    window = window[CONTEXT:-CONTEXT]
    c = encode(window)
    exon, intron = c[:DONOR_UP], c[DONOR_UP:]
    extra = [kmer_freq(exon, 3), kmer_freq(intron, 3),
             np.array([np.mean(np.isin(exon[-20:], (1, 2))), np.mean(np.isin(intron[2:22], (1, 2)))],
                      dtype=np.float32)]
    return np.concatenate([_onehot(c), *extra, _context_features(left), _context_features(right)])


def acceptor_features(window: str) -> np.ndarray:
    """window: CONTEXT | intron(60) | AG + exon(38) | CONTEXT."""
    left, right = window[:CONTEXT], window[-CONTEXT:]
    window = window[CONTEXT:-CONTEXT]
    c = encode(window)
    intron, exon = c[:ACCEPTOR_UP], c[ACCEPTOR_UP:]
    ppt = intron[-25:]                                     # polypyrimidine tract region
    bp = intron[-45:-15]                                   # branch point region
    ag_zone = intron[-30:]
    ag_count = sum(1 for i in range(len(ag_zone) - 1) if ag_zone[i] == 0 and ag_zone[i + 1] == 2)
    extra = [kmer_freq(intron, 3), kmer_freq(exon, 3),
             np.array([np.mean(np.isin(ppt, (1, 3))), np.mean(ppt == 3), np.mean(np.isin(bp, (1, 3))),
                       ag_count, np.mean(np.isin(exon, (1, 2)))], dtype=np.float32)]
    return np.concatenate([_onehot(c), *extra, _context_features(left), _context_features(right)])


def candidate_sites(seq: str, kind: str) -> list[int]:
    """0-based offsets of every GT (donor) or AG (acceptor) with a full window inside seq."""
    s = seq.upper()
    di, up, down = ("GT", DONOR_UP, DONOR_DOWN) if kind == "donor" else ("AG", ACCEPTOR_UP, ACCEPTOR_DOWN)
    out, i = [], s.find(di)
    while i != -1:
        if i - up >= 0 and i + down <= len(s):
            out.append(i)
        i = s.find(di, i + 1)
    return out


def site_window(seq: str, pos: int, kind: str) -> str:
    """Core window plus CONTEXT on each side; bases beyond the ends of seq are filled with N."""
    up, down = (DONOR_UP, DONOR_DOWN) if kind == "donor" else (ACCEPTOR_UP, ACCEPTOR_DOWN)
    a, b = pos - up - CONTEXT, pos + down + CONTEXT
    core = seq[max(0, a):min(len(seq), b)].upper()
    return "N" * max(0, -a) + core + "N" * max(0, b - len(seq))


# ---------------------------------------------------------------------------------------------
# Coding potential
# ---------------------------------------------------------------------------------------------
_STOPS = {48, 50, 56}          # TAA, TAG, TGA as base-4 codon indices (T=3, A=0, G=2)


def _codons(codes: np.ndarray, frame: int) -> np.ndarray:
    sub = codes[frame:]
    n = len(sub) // 3
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    c = sub[:n * 3].reshape(n, 3).astype(np.int64)
    bad = (c == 4).any(axis=1)
    idx = c[:, 0] * 16 + c[:, 1] * 4 + c[:, 2]
    idx[bad] = -1
    return idx


def _fickett(codes: np.ndarray) -> float:
    """Fickett TESTCODE statistic (Fickett 1982), a classic frame-free coding measure."""
    pos_prob = {0: [.94, .68, .84, .93, .58, .68, .45, .34, .20, .22],
                1: [.80, .70, .70, .81, .66, .48, .51, .33, .30, .23],
                2: [.90, .88, .74, .64, .53, .48, .27, .16, .08, .08],
                3: [.97, .97, .91, .68, .69, .44, .54, .20, .09, .09]}
    pos_w = [.26, .18, .31, .33]
    con_prob = {0: [.28, .49, .44, .55, .62, .49, .67, .65, .81, .21],
                1: [.82, .64, .51, .64, .59, .59, .43, .44, .39, .31],
                2: [.40, .54, .47, .64, .64, .73, .41, .41, .33, .29],
                3: [.28, .24, .39, .40, .55, .75, .56, .69, .51, .58]}
    con_w = [.11, .12, .15, .09]
    pos_cut = [1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 0.0]
    con_cut = [.33, .31, .29, .27, .25, .23, .21, .19, .17, 0]
    valid = codes[codes < 4]
    if len(valid) < 6:
        return 0.0
    score = 0.0
    for b in range(4):
        counts = [np.sum(codes[f::3] == b) for f in range(3)]
        pos_val = max(counts) / (min(counts) + 1.0)
        con_val = np.sum(codes == b) / len(valid)
        pi = next(i for i, c in enumerate(pos_cut) if pos_val >= c)
        ci = next(i for i, c in enumerate(con_cut) if con_val >= c)
        score += pos_prob[b][pi] * pos_w[b] + con_prob[b][ci] * con_w[b]
    return float(score)


def _period3(codes: np.ndarray) -> float:
    """Relative Fourier power at period 3, summed over the four base indicator sequences."""
    n = len(codes)
    if n < 9:
        return 0.0
    k = n // 3
    total, avg = 0.0, 0.0
    for b in range(4):
        x = (codes == b).astype(np.float32)
        f = np.abs(np.fft.rfft(x - x.mean())) ** 2
        total += f[k] if k < len(f) else 0.0
        avg += f[1:].mean() if len(f) > 1 else 0.0
    return float(total / (avg + 1e-6))


def coding_features(seq: str) -> np.ndarray:
    c = encode(seq)
    frames = []
    for f in range(3):
        cod = _codons(c, f)
        cod = cod[cod >= 0]
        stops = sum(int(x in _STOPS) for x in cod)
        usage = np.bincount(cod, minlength=64).astype(np.float32) / max(1, len(cod))
        pos = np.zeros(12, dtype=np.float32)                 # base composition at codon positions 1-3
        sub = c[f:]
        for p in range(3):
            col = sub[p::3]
            col = col[col < 4]
            if len(col):
                pos[p * 4:(p + 1) * 4] = np.bincount(col, minlength=4)[:4] / len(col)
        frames.append((stops / max(1, len(cod)), -pos[2], usage, pos))   # fewest stops, then most G at pos 1
    frames.sort(key=lambda t: (t[0], t[1]))
    best = frames[0]
    stop_rates = np.array([fr[0] for fr in frames], dtype=np.float32)
    gc = np.mean(np.isin(c[c < 4], (1, 2))) if np.any(c < 4) else 0.0
    di = kmer_freq(c, 2)
    cpg_oe = di[6] / max(1e-6, (np.mean(c == 1) * np.mean(c == 2)))
    return np.concatenate([stop_rates, best[2], best[3], frames[1][3], kmer_freq(c, 1), di, kmer_freq(c, 3),
                           np.array([_fickett(c), _period3(c), gc, cpg_oe], dtype=np.float32)]).astype(np.float32)


FEATURE_NAMES: dict[str, list[str]] = {}


def _names():
    b = "ACGT"
    k3 = [x + y + z for x in b for y in b for z in b]
    k2 = [x + y for x in b for y in b]
    ctx = [f"{side}_context_{n}" for side in ("upstream", "downstream")
           for n in [f"tri_{k}" for k in k3] + ["stop_rate_min", "stop_rate_mid", "stop_rate_max", "gc", "known"]]
    donor = [f"pos{p - DONOR_UP:+d}_{n}" for p in range(DONOR_UP + DONOR_DOWN) for n in b] + \
        [f"exon_{k}" for k in k3] + [f"intron_{k}" for k in k3] + ["exon_GC_last20", "intron_GC_3to22"] + ctx
    acc = [f"pos{p - ACCEPTOR_UP:+d}_{n}" for p in range(ACCEPTOR_UP + ACCEPTOR_DOWN) for n in b] + \
        [f"intron_{k}" for k in k3] + [f"exon_{k}" for k in k3] + \
        ["ppt_pyrimidine", "ppt_T", "branch_region_pyrimidine", "AG_in_last30", "exon_GC"] + ctx
    coding = ["stop_rate_frame1", "stop_rate_frame2", "stop_rate_frame3"] + [f"codon_{k}" for k in k3] + \
        [f"best_frame_pos{p + 1}_{n}" for p in range(3) for n in b] + \
        [f"second_frame_pos{p + 1}_{n}" for p in range(3) for n in b] + \
        [f"freq_{n}" for n in b] + [f"di_{k}" for k in k2] + [f"tri_{k}" for k in k3] + \
        ["fickett", "period3_power", "gc", "cpg_obs_exp"]
    FEATURE_NAMES.update(donor=donor, acceptor=acc, coding=coding)


_names()


# ---------------------------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def model_card() -> dict:
    p = MODEL_DIR / "model_card.json"
    return json.loads(p.read_text()) if p.exists() else {}


@lru_cache(maxsize=4)
def _session(name: str):
    import onnxruntime as ort
    path = MODEL_DIR / f"{name}.onnx"
    if not path.exists():
        return None
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3          # the converter declares the label output as shape [1]; harmless
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def available() -> bool:
    return all((MODEL_DIR / f"{n}.onnx").exists() for n in ("splice_donor", "splice_acceptor", "coding_potential"))


def predict(name: str, X: np.ndarray) -> np.ndarray:
    """Probability of the positive class for each row."""
    sess = _session(name)
    if sess is None or len(X) == 0:
        return np.zeros(len(X), dtype=np.float32)
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: np.asarray(X, dtype=np.float32)})
    probs = out[1]
    if isinstance(probs, list):                                  # zipmap output: list of dicts
        return np.array([p[1] for p in probs], dtype=np.float32)
    return np.asarray(probs)[:, 1].astype(np.float32)


def threshold(name: str) -> float:
    return float(model_card().get("models", {}).get(name, {}).get("threshold", 0.5))


def score_sites(seq: str, kind: str, positions: list[int] | None = None) -> tuple[list[int], np.ndarray]:
    """Scores for candidate donor or acceptor sites in seq (transcript orientation)."""
    pos = candidate_sites(seq, kind) if positions is None else positions
    if not pos:
        return [], np.zeros(0, dtype=np.float32)
    fx = donor_features if kind == "donor" else acceptor_features
    X = np.vstack([fx(site_window(seq, p, kind)) for p in pos])
    return pos, predict(f"splice_{kind}", X)


def coding_probability(seq: str) -> float:
    s = "".join(ch for ch in seq.upper() if ch in "ACGTN")
    if len(s) < 60:
        return float("nan")
    if len(s) <= CODING_WINDOW * 2:
        return float(predict("coding_potential", coding_features(s)[None, :])[0])
    starts = range(0, len(s) - CODING_WINDOW + 1, CODING_WINDOW // 2)
    X = np.vstack([coding_features(s[i:i + CODING_WINDOW]) for i in starts])
    return float(np.mean(predict("coding_potential", X)))


def coding_track(seq: str, step: int | None = None) -> tuple[list[int], np.ndarray]:
    s = "".join(ch for ch in seq.upper() if ch in "ACGTN")
    if len(s) < CODING_WINDOW:
        return [], np.zeros(0, dtype=np.float32)
    step = step or max(15, (len(s) - CODING_WINDOW) // 300 or 15)
    starts = list(range(0, len(s) - CODING_WINDOW + 1, step))
    X = np.vstack([coding_features(s[i:i + CODING_WINDOW]) for i in starts])
    return [i + CODING_WINDOW // 2 + 1 for i in starts], predict("coding_potential", X)


def site_profile(ref: str, mut: str, var_start: int, var_end_ref: int, var_end_mut: int,
                 radius: int = 50) -> dict:
    """Donor and acceptor scores of every candidate site near a variant, before and after it.

    Returns {kind: {ref_position: (ref_score, mut_score)}}, positions 0-based in ref. A site that only
    exists in one sequence scores 0 in the other."""
    shift = var_end_mut - var_end_ref
    out = {}
    for kind in ("donor", "acceptor"):
        lo, hi = var_start - radius, var_end_ref + radius
        rpos = [p for p in candidate_sites(ref, kind) if lo <= p <= hi]
        mpos = [p for p in candidate_sites(mut, kind) if lo <= p <= hi + shift]
        rp, rs = score_sites(ref, kind, rpos) if rpos else ([], np.zeros(0))
        mp, ms = score_sites(mut, kind, mpos) if mpos else ([], np.zeros(0))
        rmap = {p: float(s) for p, s in zip(rp, rs)}
        span_r, span_m = max(1, var_end_ref - var_start), max(1, var_end_mut - var_start)

        def to_ref(p):
            if p < var_start:
                return p
            if p >= var_end_mut:
                return p - shift
            # a site inside the changed bases: place it proportionally within the reference block
            return var_start + round((p - var_start) * span_r / span_m, 3) + (0.001 if shift > 0 else 0)
        mmap = {to_ref(p): float(s) for p, s in zip(mp, ms)}
        out[kind] = {p: (rmap.get(p, 0.0), mmap.get(p, 0.0)) for p in sorted(set(rmap) | set(mmap))}
    return out


def splice_impact(ref: str, mut: str, var_start: int, var_end_ref: int, var_end_mut: int,
                  radius: int = 50) -> dict:
    """Largest gain and loss of donor and acceptor scores around a variant (SpliceAI-style deltas)."""
    prof = site_profile(ref, mut, var_start, var_end_ref, var_end_mut, radius)
    out = {}
    for kind, sites in prof.items():
        best_gain, best_loss = (0.0, None), (0.0, None)
        for p, (r, m) in sites.items():
            d = m - r
            if d > best_gain[0]:
                best_gain = (d, p)
            if d < best_loss[0]:
                best_loss = (d, p)
        out[kind] = {"gain": round(best_gain[0], 3), "gain_pos": best_gain[1],
                     "loss": round(best_loss[0], 3), "loss_pos": best_loss[1]}
    out["max_delta"] = max(abs(out[k][x]) for k in ("donor", "acceptor") for x in ("gain", "loss"))
    out["profile"] = prof
    return out
