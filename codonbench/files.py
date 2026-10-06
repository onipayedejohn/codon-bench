"""Read uploaded files safely: check what a file really is, then stream its sequences.

Large files (up to the upload limit in config.py) are never loaded whole. FASTA is read line by line;
letter counts are kept for every record, and the sequence itself is only kept
for records short enough to search for open reading frames.
"""

from __future__ import annotations

import csv
import gzip
import io
import itertools
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import seq as S

from .config import MAX_IMAGE_MB, MAX_PDF_MB, MAX_UPLOAD_MB  # noqa: E402

MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024
KEEP_LIMIT = 200_000            # records up to this length keep their sequence (for ORFs and protein stats)
MAX_ROWS_SHOWN = 2000

SEQ_EXT = {".fasta", ".fa", ".fna", ".ffn", ".frn", ".faa", ".fas", ".seq", ".txt", ".gb", ".gbk", ".gbff",
           ".genbank"}
VAR_EXT = {".vcf"}
TABLE_EXT = {".csv", ".tsv", ".xlsx"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DOC_EXT = {".pdf"}
ACCEPTED = SEQ_EXT | TABLE_EXT | IMAGE_EXT | DOC_EXT | VAR_EXT
ACCEPT_ATTR = sorted(ACCEPTED | {".gz"})


class Blocked(Exception):
    """The file or input is not something this app analyses. The message says why."""


@dataclass
class FileKind:
    family: str        # "sequence", "table", "image", "pdf"
    ext: str
    gz: bool = False


def sniff(path: str | Path, name: str, size: int) -> FileKind:
    """Decide what a file is from its name AND its first bytes; refuse anything else."""
    if size > MAX_BYTES:
        raise Blocked(f"The file is {size / 1048576:.0f} MB. The limit is {MAX_UPLOAD_MB} MB. Compress it with gzip "
                      "(.gz) or split it into smaller files.")
    if size == 0:
        raise Blocked("The file is empty.")
    lower = name.lower()
    gz = lower.endswith(".gz")
    ext = Path(lower[:-3] if gz else lower).suffix
    if ext not in ACCEPTED:
        raise Blocked(f"'{name}' is not a supported file type. Use FASTA (.fasta, .fa, .fna), TXT, GenBank, "
                      "CSV, TSV, Excel (.xlsx), an image (PNG, JPG, TIFF) or a PDF.")
    with open(path, "rb") as fh:
        head = fh.read(4096)
    if gz:
        if not head.startswith(b"\x1f\x8b"):
            raise Blocked("The file ends in .gz but is not gzip-compressed.")
        if ext not in SEQ_EXT | VAR_EXT | {".csv", ".tsv"}:
            raise Blocked("Only compressed sequence or CSV files (.fa.gz, .fasta.gz, .csv.gz) are supported.")
        with gzip.open(path, "rb") as g:
            head = g.read(4096)
    family = ("sequence" if ext in SEQ_EXT else "variants" if ext in VAR_EXT else "table" if ext in TABLE_EXT else
              "image" if ext in IMAGE_EXT else "pdf")
    if family == "image" and size > MAX_IMAGE_MB * 1048576:
        raise Blocked(f"Images can be up to {MAX_IMAGE_MB} MB. This one is {size / 1048576:.1f} MB.")
    if family == "pdf" and size > MAX_PDF_MB * 1048576:
        raise Blocked(f"PDFs can be up to {MAX_PDF_MB} MB. This one is {size / 1048576:.1f} MB.")
    if family == "image":
        sig = {b"\x89PNG": "png", b"\xff\xd8\xff": "jpg", b"II*\x00": "tif", b"MM\x00*": "tif", b"BM": "bmp"}
        if not (any(head.startswith(k) for k in sig) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")):
            raise Blocked(f"'{name}' says it is an image but its contents are not a PNG, JPG, TIFF, BMP or WEBP.")
    elif family == "pdf":
        if not head.startswith(b"%PDF"):
            raise Blocked(f"'{name}' says it is a PDF but its contents are not.")
    elif ext == ".xlsx":
        if not head.startswith(b"PK"):
            raise Blocked(f"'{name}' says it is an Excel file but its contents are not.")
        try:
            with zipfile.ZipFile(path) as z:
                if "xl/workbook.xml" not in z.namelist():
                    raise Blocked(f"'{name}' is a zip archive, not an Excel workbook.")
        except zipfile.BadZipFile as e:
            raise Blocked(f"'{name}' is not a valid Excel file.") from e
    else:  # text formats
        magic = {b"\x89PNG": "a PNG image", b"\xff\xd8\xff": "a JPG image", b"%PDF": "a PDF", b"PK\x03\x04": "a zip or Office file",
                 b"\x1f\x8b": "a gzip archive (name it .gz)", b"GIF8": "a GIF image", b"II*\x00": "a TIFF image",
                 b"MM\x00*": "a TIFF image", b"\xd0\xcf\x11\xe0": "an old Office file (.xls/.doc)"}
        for sig, what in magic.items():
            if head.startswith(sig):
                raise Blocked(f"'{name}' is named like a text file but is actually {what}.")
        control = sum(1 for b in head if b < 9 or 13 < b < 32 or b == 127)
        if b"\x00" in head or control > max(2, len(head) // 100):
            raise Blocked(f"'{name}' contains binary data, so it is not a text sequence file.")
    return FileKind(family, ext, gz)


def open_text(path: str | Path, gz: bool) -> io.TextIOBase:
    raw = gzip.open(path, "rb") if gz else open(path, "rb")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=None)


# ---------------------------------------------------------------------------------------------
# Streaming records
# ---------------------------------------------------------------------------------------------
_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["*"]


class Accumulator:
    """Collects one record's letters without keeping more than KEEP_LIMIT of them."""

    def __init__(self, rid: str, desc: str):
        self.id, self.desc = rid, desc
        self.counts = dict.fromkeys(_LETTERS, 0)
        self.length = 0
        self.other = 0
        self.parts: list[str] | None = []
        self._buf: list[str] = []
        self._buf_len = 0

    def add(self, line: str) -> None:
        body = re.sub(r"[\s\d\-\.~]", "", line).upper()
        if not body:
            return
        self._buf.append(body)
        self._buf_len += len(body)
        if self._buf_len >= 1_000_000:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        chunk = "".join(self._buf)
        self._buf, self._buf_len = [], 0
        known = 0
        for ch in _LETTERS:
            k = chunk.count(ch)
            self.counts[ch] += k
            known += k
        self.other += len(chunk) - known
        self.length += len(chunk)
        if self.parts is not None:
            if self.length <= KEEP_LIMIT:
                self.parts.append(chunk)
            else:
                self.parts = None

    def finish(self) -> "Record":
        self._flush()
        seq = "".join(self.parts) if self.parts is not None else None
        return Record(self.id, self.desc, self.length, dict(self.counts), self.other, seq)


@dataclass
class Record:
    id: str
    desc: str
    length: int
    counts: dict
    other: int            # characters that are not letters
    seq: str | None       # None when the record was too long to keep


def kind_from_counts(counts: dict, length: int, other: int, sample: str | None) -> tuple[str, str]:
    if length == 0:
        return "Unknown", "Empty record."
    if other:
        return "Unknown", "It contains characters that are not used in sequences."
    nuc = sum(counts[c] for c in S.IUPAC_NUC)
    t, u = counts["T"], counts["U"]
    strict = sum(counts[c] for c in "ACGN") + max(t, u)
    if nuc / length >= 0.98 and strict / length >= 0.90:
        if t and u:
            return "Unknown", "It contains both T and U."
        return ("RNA" if u else "DNA"), ""
    if sample is not None:
        return S.detect(sample)
    if sum(counts[c] for c in "BZJUOX") / length > 0.1:
        return "Unknown", "Too many ambiguous or rare letters."
    return "Protein", ""


def iter_fasta(fh: io.TextIOBase) -> Iterator[Record]:
    acc: Accumulator | None = None
    first = True
    genbank = False
    n = 0
    for line in fh:
        if first:
            first = False
            if line.startswith("LOCUS"):
                genbank = True
        if genbank:
            yield from _genbank(fh, line)
            return
        if line.startswith(">"):
            if acc is not None:
                yield acc.finish()
            n += 1
            head = line[1:].strip()
            rid, _, desc = head.partition(" ")
            acc = Accumulator(rid or f"record_{n}", desc)
        elif line.startswith(";"):
            continue
        else:
            if acc is None:
                acc = Accumulator("sequence_1", "")
            acc.add(line)
    if acc is not None:
        yield acc.finish()


def _genbank(fh, first_line: str) -> Iterator[Record]:
    name, desc, acc, in_seq = first_line.split()[1] if len(first_line.split()) > 1 else "record", "", None, False
    for line in itertools.chain([first_line], fh):
        if line.startswith("LOCUS"):
            parts = line.split()
            name = parts[1] if len(parts) > 1 else "record"
        elif line.startswith("DEFINITION"):
            desc = line[10:].strip()
        elif line.startswith("ORIGIN"):
            acc, in_seq = Accumulator(name, desc), True
        elif line.startswith("//"):
            if acc is not None:
                yield acc.finish()
            acc, in_seq = None, False
        elif in_seq and acc is not None:
            acc.add(line)
    if acc is not None:
        yield acc.finish()


_SEQ_COL = re.compile(r"^(seq|sequence|sequences|dna|rna|mrna|cdna|protein|peptide|nucleotide|nucleotides|cds|aa|"
                      r"amino_acids|aa_sequence|dna_sequence|rna_sequence|protein_sequence)$", re.I)
_ID_COL = re.compile(r"^(id|name|accession|gene|gene_name|label|sample|sample_id|seq_id|sequence_id|identifier)$", re.I)


def _pick_columns(header: list[str], rows: list[list[str]]) -> tuple[int, int | None]:
    seq_col = next((i for i, h in enumerate(header) if _SEQ_COL.match(str(h).strip())), None)
    if seq_col is None:
        best, best_score = None, 0.0
        for i in range(len(header)):
            vals = [r[i] for r in rows if i < len(r) and r[i]]
            if not vals:
                continue
            ok = sum(1 for v in vals if len(str(v)) >= 6 and S.parse(str(v)).kind != "Unknown") / len(vals)
            if ok > best_score:
                best, best_score = i, ok
        if best is None or best_score < 0.6:
            raise Blocked("No column of DNA, RNA or protein sequences was found. Name the column 'sequence' "
                          "or make sure most of its cells hold sequences.")
        seq_col = best
    id_col = next((i for i, h in enumerate(header) if _ID_COL.match(str(h).strip()) and i != seq_col), None)
    return seq_col, id_col


def iter_table(path: str | Path, kind: FileKind) -> Iterator[Record]:
    if kind.ext == ".xlsx":
        rows_iter = _xlsx_rows(path)
    else:
        with open_text(path, kind.gz) as probe:
            sample = probe.read(65536)
        fh = open_text(path, kind.gz)
        delim = "\t" if kind.ext == ".tsv" else (csv.Sniffer().sniff(sample, ",;\t").delimiter
                                                 if sample.strip() else ",")
        rows_iter = csv.reader(fh, delimiter=delim)
    try:
        header = [str(h) if h is not None else "" for h in next(rows_iter)]
    except StopIteration as e:
        raise Blocked("The table is empty.") from e
    buffered = []
    for r in rows_iter:
        buffered.append(["" if v is None else str(v) for v in r])
        if len(buffered) >= 200:
            break
    seq_col, id_col = _pick_columns(header, buffered)

    def gen():
        n = 0
        for r in buffered:
            n += 1
            yield _row_record(r, seq_col, id_col, n)
        for r in rows_iter:
            n += 1
            yield _row_record(["" if v is None else str(v) for v in r], seq_col, id_col, n)

    yield from gen()


def _row_record(r, seq_col, id_col, n) -> Record:
    text = r[seq_col] if seq_col < len(r) else ""
    rid = (r[id_col] if id_col is not None and id_col < len(r) and r[id_col] else f"row_{n}")
    acc = Accumulator(str(rid), "")
    acc.add(text)
    rec = acc.finish()
    rec.desc = "__raw__" + text[:400]       # kept so prose in a cell can be recognised
    return rec


def _xlsx_rows(path):
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    for ws in wb.worksheets:
        it = ws.iter_rows(values_only=True)
        first = next(it, None)
        if first is None:
            continue
        yield list(first)
        yield from it
        return


def records(path: str | Path, kind: FileKind) -> Iterator[Record]:
    if kind.family == "sequence":
        return iter_fasta(open_text(path, kind.gz))
    if kind.family == "table":
        return iter_table(path, kind)
    raise Blocked("Images and PDFs are read on the 'Read a document' page.")


def check_text_file(path: str | Path, kind: FileKind) -> None:
    """Refuse a text file that is ordinary writing rather than sequences."""
    with open_text(path, kind.gz) as fh:
        sample = fh.read(20000)
    if ">" in sample[:2000] or sample.startswith("LOCUS"):
        return
    if S.looks_like_prose(sample):
        raise Blocked("This text file looks like ordinary writing, not sequences. Use FASTA format or a plain "
                      "sequence.")
