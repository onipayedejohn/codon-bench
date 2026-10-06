"""Sequence biology: alphabets, cleaning, the central dogma, codons, ORFs and properties.

Everything here is plain Python plus Biopython, so it can be tested on its own
and reused outside the app. Positions shown to users are 1-based and inclusive.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp, molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd as KYTE_DOOLITTLE

# ---------------------------------------------------------------------------------------------
# Alphabets
# ---------------------------------------------------------------------------------------------
DNA_STRICT = set("ACGT")
RNA_STRICT = set("ACGU")
IUPAC_NUC = set("ACGTURYSWKMBDHVN")
AMINO_20 = set("ACDEFGHIKLMNPQRSTVWY")
PROTEIN_EXTRA = set("BZJUOX*")            # ambiguity codes, selenocysteine, pyrrolysine, stop
PROTEIN_ALL = AMINO_20 | PROTEIN_EXTRA

GENETIC_CODES = {1: "Standard", 2: "Vertebrate mitochondrial", 4: "Mycoplasma and protozoan mitochondrial",
                 5: "Invertebrate mitochondrial", 11: "Bacterial, archaeal and plant plastid"}

BASES = {
    "A": dict(name="Adenine", kind="Purine", rings=2, formula="C5H5N5", pairs_dna="T", pairs_rna="U", bonds=2,
              found="DNA and RNA"),
    "G": dict(name="Guanine", kind="Purine", rings=2, formula="C5H5N5O", pairs_dna="C", pairs_rna="C", bonds=3,
              found="DNA and RNA"),
    "C": dict(name="Cytosine", kind="Pyrimidine", rings=1, formula="C4H5N3O", pairs_dna="G", pairs_rna="G", bonds=3,
              found="DNA and RNA"),
    "T": dict(name="Thymine", kind="Pyrimidine", rings=1, formula="C5H6N2O2", pairs_dna="A", pairs_rna="A", bonds=2,
              found="DNA only"),
    "U": dict(name="Uracil", kind="Pyrimidine", rings=1, formula="C4H4N2O2", pairs_dna="A", pairs_rna="A", bonds=2,
              found="RNA only"),
}

AMINO_ACIDS = {  # one-letter: (three-letter, name, class)
    "A": ("Ala", "Alanine", "Nonpolar"), "R": ("Arg", "Arginine", "Basic"), "N": ("Asn", "Asparagine", "Polar"),
    "D": ("Asp", "Aspartic acid", "Acidic"), "C": ("Cys", "Cysteine", "Polar"), "E": ("Glu", "Glutamic acid", "Acidic"),
    "Q": ("Gln", "Glutamine", "Polar"), "G": ("Gly", "Glycine", "Nonpolar"), "H": ("His", "Histidine", "Basic"),
    "I": ("Ile", "Isoleucine", "Nonpolar"), "L": ("Leu", "Leucine", "Nonpolar"), "K": ("Lys", "Lysine", "Basic"),
    "M": ("Met", "Methionine", "Nonpolar"), "F": ("Phe", "Phenylalanine", "Nonpolar"),
    "P": ("Pro", "Proline", "Nonpolar"), "S": ("Ser", "Serine", "Polar"), "T": ("Thr", "Threonine", "Polar"),
    "W": ("Trp", "Tryptophan", "Nonpolar"), "Y": ("Tyr", "Tyrosine", "Polar"), "V": ("Val", "Valine", "Nonpolar"),
    "*": ("Stop", "Stop codon", "Stop"), "X": ("Xaa", "Unknown", "Unknown"),
    "U": ("Sec", "Selenocysteine", "Polar"), "O": ("Pyl", "Pyrrolysine", "Polar"),
    "B": ("Asx", "Asn or Asp", "Unknown"), "Z": ("Glx", "Gln or Glu", "Unknown"), "J": ("Xle", "Leu or Ile", "Unknown"),
}


# ---------------------------------------------------------------------------------------------
# Cleaning and detection
# ---------------------------------------------------------------------------------------------
@dataclass
class Parsed:
    seq: str                      # cleaned, upper case
    kind: str                     # "DNA", "RNA", "Protein" or "Unknown"
    header: str = ""
    ambiguous: int = 0            # IUPAC ambiguity codes (N, R, Y, ...) or X in proteins
    removed: int = 0              # characters dropped while cleaning (digits, spaces, gaps)
    reason: str = ""              # why it is Unknown


def clean(text: str) -> tuple[str, str, int]:
    """Strip FASTA/GenBank decoration. Returns (sequence, first header, characters removed)."""
    header, keep, removed = "", [], 0
    lines = text.splitlines()
    if any(ln.startswith("ORIGIN") for ln in lines):           # GenBank record: sequence follows ORIGIN
        locus = next((ln for ln in lines if ln.startswith("DEFINITION")),
                     next((ln for ln in lines if ln.startswith("LOCUS")), ""))
        header = re.sub(r"^(DEFINITION|LOCUS)\s+", "", locus).strip()
        start = next(i for i, ln in enumerate(lines) if ln.startswith("ORIGIN"))
        lines = lines[start + 1:]
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith(">") or s.startswith(";"):
            if not header:
                header = s.lstrip(">;").strip()
            continue
        if s.upper().startswith(("ORIGIN", "LOCUS", "//")):
            continue
        body = re.sub(r"[\s\d\-\.~]", "", s)     # spaces, position numbers, alignment gaps
        removed += len(s) - len(body)
        keep.append(body)
    return "".join(keep).upper(), header, removed


# Background letter frequencies, used to tell a protein from ordinary words.
_AA_FREQ = dict(A=.0825, R=.0553, N=.0406, D=.0546, C=.0137, Q=.0393, E=.0675, G=.0707, H=.0227, I=.0596,
                L=.0966, K=.0584, M=.0242, F=.0386, P=.0470, S=.0656, T=.0534, W=.0108, Y=.0292, V=.0687)
_EN_FREQ = dict(A=.082, B=.015, C=.028, D=.043, E=.127, F=.022, G=.020, H=.061, I=.070, J=.0015, K=.0077, L=.040,
                M=.024, N=.067, O=.075, P=.019, Q=.00095, R=.060, S=.063, T=.091, U=.028, V=.0098, W=.024, X=.0015,
                Y=.020, Z=.00074)


def protein_vs_english(seq: str) -> float:
    """Mean log-likelihood ratio per letter: above 0 looks like a protein, below 0 like English."""
    letters = [c for c in seq if c in _AA_FREQ]
    if not letters:
        return -9.0
    return sum(math.log(_AA_FREQ[c] / _EN_FREQ[c]) for c in letters) / len(letters)


_COMMON_WORDS = set("""the of and to in is was for on that with as by are at be this from it an or were which
not have has had but can all their its also been more one two may than other into these such after between
our we they he she his her you your i patients patient results study data total sales revenue report name age
sex date page table figure increased decreased using used new number please dear hello thank thanks regards
about over under per when where what who how there here will would could should only very most each both""".split())


def looks_like_prose(text: str) -> bool:
    """True for ordinary writing: common English words, or Capitalised words mixed with lower case."""
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith((">", ";"))]
    tokens = re.findall(r"[A-Za-z]+", " ".join(lines))
    if len(tokens) < 2:
        return False
    common = sum(t.lower() in _COMMON_WORDS for t in tokens)
    title = sum(1 for t in tokens if len(t) > 1 and t[0].isupper() and t[1:].islower())
    return common / len(tokens) >= 0.15 or title >= 2 or (title >= 1 and common >= 1)


def detect(seq: str, raw: str | None = None) -> tuple[str, str]:
    """Classify a cleaned sequence. Returns (kind, reason if Unknown)."""
    n = len(seq)
    if n == 0:
        return "Unknown", "No sequence letters were found."
    counts = Counter(seq)
    nuc = sum(counts[c] for c in IUPAC_NUC)
    if nuc == n and counts["T"] and counts["U"]:
        return "Unknown", "It contains both T and U, so it is neither clean DNA nor clean RNA."
    strict_t = sum(counts[c] for c in "ACGTN")
    strict_u = sum(counts[c] for c in "ACGUN")
    if nuc / n >= 0.98 and max(strict_t, strict_u) / n >= 0.90:
        if counts["U"] and not counts["T"]:
            return "RNA", ""
        if counts["U"] and counts["T"]:
            return "Unknown", "It contains both T and U, so it is neither clean DNA nor clean RNA."
        return "DNA", ""
    if raw is not None and looks_like_prose(raw):
        return "Unknown", "It looks like ordinary writing, not a DNA, RNA or protein sequence."
    bad = sorted({c for c in counts if c not in PROTEIN_ALL})
    if bad:
        shown = "".join(bad[:8])
        return "Unknown", f"It contains characters that are not used in DNA, RNA or protein sequences ({shown})."
    if n < 5:
        return "Unknown", "It is too short to tell what it is."
    rare = sum(counts[c] for c in "BZJUOX")
    if rare / n > 0.1:
        return "Unknown", "Too many letters are ambiguous or rare amino acid codes to be a real protein."
    if protein_vs_english(seq) < 0.05:
        return "Unknown", "The letters look like ordinary words, not an amino acid sequence."
    return "Protein", ""


def parse(text: str) -> Parsed:
    seq, header, removed = clean(text)
    kind, reason = detect(seq, text)
    if kind in ("DNA", "RNA"):
        amb = sum(1 for c in seq if c not in "ACGTU")
    elif kind == "Protein":
        amb = sum(1 for c in seq if c not in AMINO_20 and c != "*")
    else:
        amb = 0
    return Parsed(seq, kind, header, amb, removed, reason)


# ---------------------------------------------------------------------------------------------
# Central dogma
# ---------------------------------------------------------------------------------------------
def to_dna(s: str) -> str:
    return s.replace("U", "T")


def to_rna(s: str) -> str:
    return s.replace("T", "U")


def complement(s: str, rna: bool = False) -> str:
    out = str(Seq(to_dna(s)).complement())
    return to_rna(out) if rna else out


def reverse_complement(s: str, rna: bool = False) -> str:
    out = str(Seq(to_dna(s)).reverse_complement())
    return to_rna(out) if rna else out


def transcribe(dna: str, strand: str = "coding") -> str:
    """mRNA 5'->3'. strand="coding": the given 5'->3' strand has the same sequence as the mRNA.
    strand="template": the given 5'->3' strand is the one RNA polymerase reads."""
    dna = to_dna(dna)
    coding = dna if strand == "coding" else reverse_complement(dna)
    return to_rna(coding)


def reverse_transcribe(rna: str) -> dict[str, str]:
    """Reverse transcriptase makes first-strand cDNA (complementary, written 5'->3'),
    then a second strand that matches the RNA with T in place of U."""
    rna = to_rna(rna)
    first = reverse_complement(rna)
    second = to_dna(rna)
    return {"rna": rna, "cdna_first_strand": first, "cdna_second_strand": second}


def replicate(dna: str) -> dict[str, str]:
    """Semi-conservative replication: each parent strand templates a new complementary strand."""
    dna = to_dna(dna)
    return {"parent_top": dna, "parent_bottom_3to5": complement(dna), "new_on_top": complement(dna),
            "new_on_bottom": dna}


def codon_table(table_id: int = 1):
    return CodonTable.unambiguous_dna_by_id[table_id]


def anticodon(codon_rna: str) -> dict[str, str]:
    """tRNA anticodon for an mRNA codon. Pairing is antiparallel, so the anticodon read 3'->5'
    is the complement of the codon, and 5'->3' it is the reverse complement."""
    c = to_rna(codon_rna)
    comp = complement(c, rna=True)
    return {"three_to_five": comp, "five_to_three": comp[::-1]}


@dataclass
class Codon:
    index: int          # 1-based codon number
    start: int          # 1-based position of the first base in the mRNA
    codon: str          # mRNA codon (U)
    anticodon_3to5: str
    anticodon_5to3: str
    aa: str             # one-letter, "*" for stop, "X" for unknown
    aa3: str
    name: str
    aa_class: str


def translate_codons(mrna: str, frame: int = 0, table_id: int = 1, from_start: bool = False,
                     to_stop: bool = True, limit: int | None = None) -> list[Codon]:
    """Read an mRNA codon by codon. from_start begins at the first AUG in the frame."""
    mrna = to_rna(mrna)
    tab = codon_table(table_id)
    begin = frame
    if from_start:
        begin = -1
        for i in range(frame, len(mrna) - 2, 3):
            if mrna[i:i + 3] == "AUG":
                begin = i
                break
        if begin < 0:
            return []
    out = []
    for k, i in enumerate(range(begin, len(mrna) - 2, 3), start=1):
        cod = mrna[i:i + 3]
        dna_cod = to_dna(cod)
        if dna_cod in tab.stop_codons:
            aa = "*"
        else:
            aa = tab.forward_table.get(dna_cod, "X")
        a3, name, cls = AMINO_ACIDS.get(aa, ("Xaa", "Unknown", "Unknown"))
        ac = anticodon(cod)
        out.append(Codon(k, i + 1, cod, ac["three_to_five"], ac["five_to_three"], aa, a3, name, cls))
        if (to_stop and aa == "*") or (limit and len(out) >= limit):
            break
    return out


def translate(nuc: str, frame: int = 0, table_id: int = 1, to_stop: bool = False) -> str:
    dna = to_dna(nuc)[frame:]
    dna = dna[: len(dna) - len(dna) % 3]
    if not dna:
        return ""
    return str(Seq(dna).translate(table=table_id, to_stop=to_stop))


# ---------------------------------------------------------------------------------------------
# Open reading frames
# ---------------------------------------------------------------------------------------------
@dataclass
class ORF:
    frame: str          # "+1", "+2", "+3", "-1", "-2", "-3"
    start: int          # 1-based, on the given (forward) strand, always <= end
    end: int
    length_nt: int
    length_aa: int      # without the stop codon
    protein: str
    complete: bool      # True if it ends in a stop codon


def find_orfs(nuc: str, table_id: int = 1, min_aa: int = 30, alt_starts: bool = False,
              include_partial: bool = False, max_orfs: int = 5000) -> list[ORF]:
    """Six-frame ORF search: start codon to the first in-frame stop codon."""
    dna = to_dna(nuc)
    n = len(dna)
    tab = codon_table(table_id)
    starts = set(tab.start_codons) if alt_starts else {"ATG"}
    stops = set(tab.stop_codons)
    out: list[ORF] = []
    for strand, s in ((1, dna), (-1, reverse_complement(dna))):
        for f in range(3):
            i = f
            open_at = -1
            while i <= n - 3:
                cod = s[i:i + 3]
                if open_at < 0 and cod in starts:
                    open_at = i
                elif open_at >= 0 and cod in stops:
                    _add(out, s, open_at, i + 3, strand, f, n, table_id, min_aa, True)
                    open_at = -1
                i += 3
            if include_partial and open_at >= 0:
                end = open_at + (n - open_at) // 3 * 3
                _add(out, s, open_at, end, strand, f, n, table_id, min_aa, False)
            if len(out) > max_orfs:
                break
    out.sort(key=lambda o: -o.length_aa)
    return out[:max_orfs]


def _add(out, s, a, b, strand, f, n, table_id, min_aa, complete):
    prot = str(Seq(s[a:b]).translate(table=table_id, to_stop=True))
    if complete is False:
        prot = str(Seq(s[a:b]).translate(table=table_id)).rstrip("*")
    # the first codon is always read as methionine, whatever the start codon
    prot = "M" + prot[1:] if prot else prot
    if len(prot) < min_aa:
        return
    if strand == 1:
        start, end = a + 1, b
    else:
        start, end = n - b + 1, n - a
    out.append(ORF(f"{'+' if strand == 1 else '-'}{f + 1}", start, end, b - a, len(prot), prot, complete))


# ---------------------------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------------------------
def nucleotide_stats(seq: str, kind: str) -> dict:
    s = seq
    n = len(s)
    c = Counter(s)
    t_or_u = "U" if kind == "RNA" else "T"
    gc = c["G"] + c["C"]
    at = c["A"] + c[t_or_u]
    known = gc + at
    stats = {
        "length": n,
        "counts": {b: c[b] for b in ("A", "C", "G", t_or_u)},
        "ambiguous": n - known,
        "gc_percent": 100 * gc / known if known else 0.0,
        "at_percent": 100 * at / known if known else 0.0,
        "purines": c["A"] + c["G"],
        "pyrimidines": c["C"] + c[t_or_u],
        "gc_skew": (c["G"] - c["C"]) / gc if gc else 0.0,
        "cpg": s.count("CG"),
    }
    try:
        stats["mw_single"] = molecular_weight(Seq(s.replace("N", "")), seq_type="RNA" if kind == "RNA" else "DNA")
    except ValueError:
        stats["mw_single"] = float("nan")
    if kind == "DNA" and known >= 2:
        clean = "".join(ch for ch in s if ch in "ACGT")
        if len(clean) < 14:
            stats["tm"] = MeltingTemp.Tm_Wallace(clean)
            stats["tm_method"] = "Wallace rule, 2(A+T) + 4(G+C)"
        else:
            stats["tm"] = MeltingTemp.Tm_GC(clean, Na=50)
            stats["tm_method"] = "GC-content formula at 50 mM Na+"
    return stats


def sliding_gc(seq: str, window: int | None = None, max_points: int = 600) -> tuple[list[int], list[float], int]:
    n = len(seq)
    if n == 0:
        return [], [], 0
    if window is None:
        window = max(10, min(500, n // 10 or 10))
    window = min(window, n)
    step = max(1, (n - window) // max_points or 1)
    xs, ys = [], []
    for i in range(0, n - window + 1, step):
        w = seq[i:i + window]
        known = sum(1 for ch in w if ch in "ACGTU")
        gc = w.count("G") + w.count("C")
        xs.append(i + window // 2 + 1)
        ys.append(100 * gc / known if known else 0.0)
    return xs, ys, window


def codon_usage(mrna: str, frame: int = 0) -> Counter:
    r = to_rna(mrna)
    return Counter(r[i:i + 3] for i in range(frame, len(r) - 2, 3) if set(r[i:i + 3]) <= RNA_STRICT)


@dataclass
class ProteinStats:
    length: int
    mw: float
    pi: float
    charge_ph7: float
    gravy: float
    instability: float
    aromaticity: float
    extinction_reduced: int
    extinction_cystines: int
    helix: float
    turn: float
    sheet: float
    composition: dict = field(default_factory=dict)


def protein_stats(prot: str) -> ProteinStats | None:
    p = "".join(ch for ch in prot.upper() if ch in AMINO_20)
    if len(p) < 2:
        return None
    pa = ProteinAnalysis(p)
    h, t, s = pa.secondary_structure_fraction()
    ext = pa.molar_extinction_coefficient()
    return ProteinStats(len(p), pa.molecular_weight(), pa.isoelectric_point(), pa.charge_at_pH(7.0), pa.gravy(),
                        pa.instability_index(), pa.aromaticity(), ext[0], ext[1], h, t, s,
                        dict(Counter(p)))


def hydropathy(prot: str, window: int = 9) -> tuple[list[int], list[float]]:
    p = [ch for ch in prot.upper() if ch in KYTE_DOOLITTLE]
    if len(p) < window:
        return [], []
    xs, ys = [], []
    for i in range(len(p) - window + 1):
        xs.append(i + window // 2 + 1)
        ys.append(sum(KYTE_DOOLITTLE[c] for c in p[i:i + window]) / window)
    return xs, ys


def codons_for(aa: str, table_id: int = 1) -> list[str]:
    tab = codon_table(table_id)
    if aa == "*":
        return [to_rna(c) for c in tab.stop_codons]
    return sorted(to_rna(c) for c, a in tab.forward_table.items() if a == aa)


def back_translation_count(prot: str, table_id: int = 1) -> tuple[int, float]:
    """How many different mRNAs could encode this protein (the genetic code is degenerate)."""
    log10 = 0.0
    total = 1
    for a in prot.upper():
        k = len(codons_for(a, table_id)) if a in AMINO_20 else 0
        if k:
            log10 += math.log10(k)
            if log10 < 300:
                total *= k
    return (total if log10 < 300 else -1), log10
