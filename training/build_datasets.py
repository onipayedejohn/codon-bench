"""Build labelled datasets from genome + GENCODE annotation.

Splice sites
  positives: every canonical GT-AG intron boundary of every annotated transcript
  negatives: GT (donor) or AG (acceptor) dinucleotides inside the same genes, on the same strand,
             that are not an annotated boundary of any transcript (sampled, 25 per positive)
Coding potential (150 nt windows)
  positives: windows of the spliced coding sequence (CDS) of one transcript per protein-coding gene,
             at random offsets, so all three reading frames appear
  negatives: introns, intergenic DNA (either strand), UTRs and spliced lncRNA exons
Transcript-level coding test (human only)
  full CDS of protein-coding genes vs full spliced lncRNA transcripts

Run from the project folder:  python training/build_datasets.py
"""
from __future__ import annotations

import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from codonbench.ml import (ACCEPTOR_DOWN, ACCEPTOR_UP, CODING_WINDOW, CONTEXT, DONOR_DOWN, DONOR_UP,  # noqa: E402
                           revcomp)

HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "raw", HERE / "processed"
NEG_PER_POS = 25
RNG = np.random.default_rng(20261006)
ATTR = re.compile(r'(\S+) "([^"]*)"')


def read_fasta(path: Path) -> dict[str, str]:
    op = gzip.open if path.suffix == ".gz" else open
    seqs, name, buf = {}, None, []
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf).upper()
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name:
        seqs[name] = "".join(buf).upper()
    return seqs


def read_gtf(path: Path) -> dict[str, dict]:
    op = gzip.open if path.suffix == ".gz" else open
    tx: dict[str, dict] = {}
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in ("exon", "CDS", "UTR"):
                continue
            a = dict(ATTR.findall(f[8]))
            t = tx.setdefault(a["transcript_id"], {
                "chrom": f[0], "strand": f[6], "gene_id": a["gene_id"], "gene_type": a.get("gene_type", ""),
                "transcript_type": a.get("transcript_type", ""), "exon": [], "CDS": [], "UTR": []})
            t[f[2]].append((int(f[3]), int(f[4])))
    for t in tx.values():
        for k in ("exon", "CDS", "UTR"):
            t[k].sort()
    return tx


def oriented(g: str, pos0: int, strand: str, up: int, down: int) -> str:
    if strand == "+":
        return g[pos0 - up: pos0 + down] if pos0 - up >= 0 else ""
    lo = pos0 - down + 1
    return revcomp(g[lo: pos0 + up + 1]) if lo >= 0 else ""


def spliced(g: str, parts: list[tuple[int, int]], strand: str) -> str:
    s = "".join(g[a - 1:b] for a, b in parts)
    return s if strand == "+" else revcomp(s)


def splice_sites(genome, tx):
    """Annotated sites as {(chrom, strand, pos0)} for donors and acceptors, plus gene spans."""
    donors, acceptors, noncanon = set(), set(), 0
    gene_sites = defaultdict(lambda: {"donor": set(), "acceptor": set()})
    spans = {}
    for t in tx.values():
        g = genome.get(t["chrom"])
        ex = t["exon"]
        if g is None:
            continue
        key = (t["chrom"], t["strand"], t["gene_id"])
        lo, hi = ex[0][0], ex[-1][1]
        s0 = spans.get(key, (lo, hi))
        spans[key] = (min(s0[0], lo), max(s0[1], hi))
        for (_, e1), (s2, _) in zip(ex, ex[1:]):
            a, b = e1 + 1, s2 - 1
            if b - a < 30:
                continue
            if t["strand"] == "+":
                d, ac = a - 1, b - 2
                ok = g[d:d + 2] == "GT" and g[ac:ac + 2] == "AG"
            else:
                d, ac = b - 1, a
                ok = g[b - 2:b] == "AC" and g[a - 1:a + 1] == "CT"
            if not ok:
                noncanon += 1
                continue
            donors.add((t["chrom"], t["strand"], d))
            acceptors.add((t["chrom"], t["strand"], ac))
            gene_sites[key]["donor"].add(d)
            gene_sites[key]["acceptor"].add(ac)
    return donors, acceptors, gene_sites, spans, noncanon


def candidates(g: str, lo: int, hi: int, strand: str, kind: str) -> np.ndarray:
    """0-based pos0 of every oriented GT/AG between 1-based lo..hi."""
    sub = g[lo - 1:hi]
    pat = {("+", "donor"): "GT", ("+", "acceptor"): "AG", ("-", "donor"): "AC", ("-", "acceptor"): "CT"}[(strand, kind)]
    idx = np.array([m.start() for m in re.finditer(f"(?={pat})", sub)], dtype=np.int64) + (lo - 1)
    return idx if strand == "+" else idx + 1


def build_splice(genome, tx, species):
    donors, acceptors, gene_sites, spans, noncanon = splice_sites(genome, tx)
    rows = {"donor": [], "acceptor": []}
    for (chrom, strand, gene), sites in gene_sites.items():
        g = genome[chrom]
        lo, hi = spans[(chrom, strand, gene)]
        for kind, up, down, allpos in (("donor", DONOR_UP + CONTEXT, DONOR_DOWN + CONTEXT, donors),
                                       ("acceptor", ACCEPTOR_UP + CONTEXT, ACCEPTOR_DOWN + CONTEXT, acceptors)):
            pos = sorted(sites[kind])
            for p in pos:
                w = oriented(g, p, strand, up, down)
                if len(w) == up + down and "N" not in w[CONTEXT:-CONTEXT]:
                    rows[kind].append((species, gene, chrom, strand, p, 1, w))
            cand = candidates(g, lo, hi, strand, kind)
            cand = np.array([c for c in cand if (chrom, strand, int(c)) not in allpos], dtype=np.int64)
            n = min(len(cand), NEG_PER_POS * max(1, len(pos)))
            for p in RNG.choice(cand, size=n, replace=False) if n else []:
                w = oriented(g, int(p), strand, up, down)
                if len(w) == up + down and "N" not in w[CONTEXT:-CONTEXT]:
                    rows[kind].append((species, gene, chrom, strand, int(p), 0, w))
    cols = ["species", "gene_id", "chrom", "strand", "pos0", "label", "window"]
    return {k: pd.DataFrame(v, columns=cols) for k, v in rows.items()}, noncanon


def _windows(s: str, n: int):
    if len(s) < CODING_WINDOW:
        return []
    starts = RNG.integers(0, len(s) - CODING_WINDOW + 1, size=n)
    return [s[i:i + CODING_WINDOW] for i in starts]


def build_coding(genome, tx, species):
    by_gene = defaultdict(list)
    for tid, t in tx.items():
        by_gene[(t["gene_id"], t["transcript_type"])].append((tid, t))
    pos, neg = [], []
    covered = defaultdict(list)
    for t in tx.values():
        covered[t["chrom"]].append((t["exon"][0][0] - 1000, t["exon"][-1][1] + 1000))
    for (gene, ttype), items in by_gene.items():
        if ttype == "protein_coding":
            tid, t = max(items, key=lambda x: sum(b - a + 1 for a, b in x[1]["CDS"]))
            g = genome.get(t["chrom"])
            if g is None or not t["CDS"]:
                continue
            cds = spliced(g, t["CDS"], t["strand"])
            for w in _windows(cds, min(12, len(cds) // CODING_WINDOW)):
                pos.append((species, gene, "CDS", 1, w))
            for (_, e1), (s2, _) in zip(t["exon"], t["exon"][1:]):
                intr = spliced(g, [(e1 + 1, s2 - 1)], t["strand"])
                for w in _windows(intr, 1 if len(intr) < 2000 else 3):
                    neg.append((species, gene, "intron", 0, w))
            if t["UTR"]:
                utr = spliced(g, t["UTR"], t["strand"])
                for w in _windows(utr, min(2, len(utr) // CODING_WINDOW)):
                    neg.append((species, gene, "UTR", 0, w))
        elif ttype == "lncRNA":
            tid, t = max(items, key=lambda x: sum(b - a + 1 for a, b in x[1]["exon"]))
            g = genome.get(t["chrom"])
            if g is None:
                continue
            s = spliced(g, t["exon"], t["strand"])
            for w in _windows(s, min(6, len(s) // CODING_WINDOW)):
                neg.append((species, gene, "lncRNA", 0, w))
    # intergenic windows, either strand
    for chrom, ivs in covered.items():
        g = genome[chrom]
        ivs.sort()
        merged = []
        for a, b in ivs:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        gaps = [(b, a2) for (_, b), (a2, _) in zip(merged, merged[1:]) if a2 - b > 5000]
        need = int(0.30 * len(pos))
        for k in range(need if gaps else 0):
            a, b = gaps[RNG.integers(len(gaps))]
            i = int(RNG.integers(a, b - CODING_WINDOW))
            w = g[i:i + CODING_WINDOW]
            if RNG.random() < 0.5:
                w = revcomp(w)
            neg.append((species, f"intergenic_{chrom}_{a}", "intergenic", 0, w))
    # balance: as many negatives as positives, mixed intron 35%, intergenic 30%, lncRNA 20%, UTR 15%
    pools = defaultdict(list)
    for r in neg:
        pools[r[2]].append(r)
    target = {"intergenic": 0.30, "lncRNA": 0.20, "UTR": 0.15}
    chosen = []
    for src, frac in target.items():
        pool = pools.get(src, [])
        k = min(len(pool), int(frac * len(pos)))
        chosen += [pool[i] for i in RNG.choice(len(pool), size=k, replace=False)] if k else []
    rest = len(pos) - len(chosen)
    pool = pools.get("intron", [])
    k = min(len(pool), rest)
    chosen += [pool[i] for i in RNG.choice(len(pool), size=k, replace=False)] if k else []
    df = pd.DataFrame(pos + chosen, columns=["species", "gene_id", "source", "label", "window"])
    df = df[~df["window"].str.contains("N")]
    return df.reset_index(drop=True)


def build_transcripts(genome, tx, species):
    rows, seen = [], set()
    for tid, t in sorted(tx.items(), key=lambda kv: -sum(b - a for a, b in kv[1]["exon"])):
        g = genome.get(t["chrom"])
        if g is None or t["gene_id"] in seen:
            continue
        if t["transcript_type"] == "protein_coding" and t["CDS"]:
            s, lab, src = spliced(g, t["CDS"], t["strand"]), 1, "CDS"
        elif t["transcript_type"] == "lncRNA":
            s, lab, src = spliced(g, t["exon"], t["strand"]), 0, "lncRNA"
        else:
            continue
        if len(s) >= 200 and "N" not in s:
            rows.append((species, t["gene_id"], tid, src, lab, s))
            seen.add(t["gene_id"])
    return pd.DataFrame(rows, columns=["species", "gene_id", "transcript_id", "source", "label", "sequence"])


def main():
    OUT.mkdir(exist_ok=True)
    stats = {}
    for species, fa, gtf in (("mouse", "mm_chr19.fa.gz", "mm_chr19.gtf.gz"), ("human", "hs_chr21.fa", "hs_chr21.gtf")):
        genome = read_fasta(RAW / fa)
        tx = read_gtf(RAW / gtf)
        sp, noncanon = build_splice(genome, tx, species)
        cod = build_coding(genome, tx, species)
        for kind, df in sp.items():
            df.to_csv(OUT / f"splice_{kind}_{species}.csv.gz", index=False)
        cod.to_csv(OUT / f"coding_{species}.csv.gz", index=False)
        if species == "human":
            build_transcripts(genome, tx, species).to_csv(OUT / "coding_transcripts_human.csv.gz", index=False)
        stats[species] = {"transcripts": len(tx), "non_canonical_introns_skipped": noncanon,
                          **{f"{k}_pos": int(v.label.sum()) for k, v in sp.items()},
                          **{f"{k}_neg": int((v.label == 0).sum()) for k, v in sp.items()},
                          "coding_pos": int(cod.label.sum()), "coding_neg": int((cod.label == 0).sum()),
                          "coding_neg_sources": cod[cod.label == 0].source.value_counts().to_dict()}
        print(species, stats[species])
    pd.Series(stats).to_json(OUT / "dataset_stats.json", indent=2)


if __name__ == "__main__":
    main()
