"""Summarise every record in an uploaded file, with a progress callback for the app."""

from __future__ import annotations

import csv
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import files as F
from . import seq as S

COLUMNS = ["id", "description", "type", "length", "gc_percent", "ambiguous", "longest_orf_aa", "orf_frame",
           "protein_mw_kda", "protein_pi", "status"]


@dataclass
class BatchResult:
    rows: list[dict]                 # first MAX_ROWS_SHOWN rows, for the table
    n_records: int
    n_ok: int
    n_blocked: int
    kinds: dict
    lengths: list[int]               # a sample, for the histogram
    gcs: list[float]
    csv_path: str
    proteins_path: str | None
    seconds: float
    total_letters: int
    notes: list[str] = field(default_factory=list)


def summarise(rec: F.Record, table_id: int = 1, min_orf_aa: int = 30) -> dict:
    raw = rec.desc[7:] if rec.desc.startswith("__raw__") else None
    desc = "" if raw is not None else rec.desc
    sample = rec.seq if rec.seq is not None else None
    kind, reason = F.kind_from_counts(rec.counts, rec.length, rec.other, sample)
    if kind == "Protein" and raw is not None and S.looks_like_prose(raw):
        kind, reason = "Unknown", "It looks like ordinary writing."
    row = dict.fromkeys(COLUMNS, "")
    row.update(id=rec.id, description=desc[:120], type=kind if kind != "Unknown" else "Blocked",
               length=rec.length)
    if kind == "Unknown":
        row["status"] = f"Not analysed: {reason}"
        return row
    c = rec.counts
    if kind in ("DNA", "RNA"):
        tu = c["U"] if kind == "RNA" else c["T"]
        gc, at = c["G"] + c["C"], c["A"] + tu
        row["gc_percent"] = round(100 * gc / (gc + at), 2) if gc + at else ""
        row["ambiguous"] = rec.length - gc - at
        if rec.seq is not None:
            orfs = S.find_orfs(rec.seq, table_id=table_id, min_aa=min_orf_aa, max_orfs=50)
            if orfs:
                row["longest_orf_aa"] = orfs[0].length_aa
                row["orf_frame"] = orfs[0].frame
                row["_protein"] = orfs[0].protein
            row["status"] = "Analysed"
        else:
            row["status"] = f"Composition only: longer than {F.KEEP_LIMIT:,} letters, so ORFs were not searched"
    else:
        row["ambiguous"] = sum(c[x] for x in "BZJUOX")
        if rec.seq is not None:
            ps = S.protein_stats(rec.seq)
            if ps:
                row["protein_mw_kda"] = round(ps.mw / 1000, 2)
                row["protein_pi"] = round(ps.pi, 2)
            row["_protein"] = rec.seq.rstrip("*")
        row["status"] = "Analysed"
    return row


def run(path: str, name: str, size: int, table_id: int = 1, min_orf_aa: int = 30,
        progress: Callable[[float, str], None] | None = None) -> BatchResult:
    kind = F.sniff(path, name, size)
    if kind.family in ("image", "pdf"):
        raise F.Blocked("Images and PDFs go on the 'Read a document' page, which reads text from them.")
    if kind.family == "variants":
        raise F.Blocked("VCF files hold variants, not sequences. Open them on the Mutations page, under "
                        "'Batch variants'.")
    if kind.family == "sequence":
        F.check_text_file(path, kind)
    t0 = time.time()
    out = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    prot = tempfile.NamedTemporaryFile("w", suffix=".faa", delete=False)
    writer = csv.DictWriter(out, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    rows, kinds, lengths, gcs = [], {}, [], []
    n = ok = blocked = letters = n_prot = 0
    last = 0.0
    for rec in F.records(path, kind):
        row = summarise(rec, table_id, min_orf_aa)
        n += 1
        letters += rec.length
        kinds[row["type"]] = kinds.get(row["type"], 0) + 1
        if row["type"] == "Blocked":
            blocked += 1
        else:
            ok += 1
            if len(lengths) < 20000:
                lengths.append(rec.length)
                if row["gc_percent"] != "":
                    gcs.append(row["gc_percent"])
        if row.get("_protein"):
            prot.write(f">{row['id']} {row['type']} longest ORF {row.get('orf_frame', '')}\n{row['_protein']}\n")
            n_prot += 1
        writer.writerow(row)
        if len(rows) < F.MAX_ROWS_SHOWN:
            rows.append({k: row[k] for k in COLUMNS})
        if progress and time.time() - last > 0.5:
            last = time.time()
            progress(min(0.95, letters / max(size, 1)), f"{n:,} records, {letters / 1e6:,.1f} million letters")
    out.close()
    prot.close()
    if n == 0:
        raise F.Blocked("No sequences were found in this file.")
    if ok == 0:
        raise F.Blocked("None of the records in this file is a DNA, RNA or protein sequence, so nothing was "
                        "analysed.")
    notes = []
    if n > F.MAX_ROWS_SHOWN:
        notes.append(f"The table shows the first {F.MAX_ROWS_SHOWN:,} of {n:,} records. The download has all of them.")
    return BatchResult(rows, n, ok, blocked, kinds, lengths, gcs, out.name, prot.name if n_prot else None,
                       time.time() - t0, letters, notes)
