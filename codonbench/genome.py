"""Reference genomes, VCF input and variant annotation for any organism.

A reference is one of
  coding   a single coding sequence that starts at c.1 (HGVS c. notation)
  genomic  any DNA sequence without annotation (HGVS g. notation, DNA-level description only)
  annotated  a GenBank record with CDS and RNA gene features (g. or m. for mitochondria); variants
           inside a CDS are translated with that gene's genetic code
"""
from __future__ import annotations

import gzip
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import mutations as M
from . import seq as S

_COMP = str.maketrans("ACGTN", "TGCAN")
TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
SPECTRUM = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
NONCODING_TYPES = ("tRNA", "rRNA", "ncRNA", "misc_RNA", "D-loop", "5'UTR", "3'UTR", "regulatory", "stem_loop",
                   "repeat_region", "misc_feature")


def rc(s: str) -> str:
    return s.translate(_COMP)[::-1]


@dataclass
class Feature:
    kind: str                       # CDS, tRNA, rRNA, ...
    name: str
    parts: list[tuple[int, int]]    # 0-based half-open, ordered 5'->3' along the feature
    strand: int                     # +1 or -1
    table: int = 1
    product: str = ""
    codon_start: int = 1

    @property
    def start(self) -> int:
        return min(a for a, _ in self.parts)

    @property
    def end(self) -> int:
        return max(b for _, b in self.parts)

    def contains(self, pos0: int) -> bool:
        return any(a <= pos0 < b for a, b in self.parts)

    def to_cds_index(self, pos0: int) -> int | None:
        """0-based index within the spliced feature (in feature orientation), or None."""
        k = 0
        for a, b in self.parts:
            if a <= pos0 < b:
                return k + (pos0 - a if self.strand == 1 else b - 1 - pos0)
            k += b - a
        return None


@dataclass
class Reference:
    id: str
    description: str
    seq: str
    features: list[Feature] = field(default_factory=list)
    kind: str = "genomic"           # coding | genomic | annotated
    prefix: str = "g"               # c, g or m
    organism: str = ""
    table: int = 1
    eukaryotic_nuclear: bool = True

    @property
    def cds(self) -> list[Feature]:
        return [f for f in self.features if f.kind == "CDS"]

    def feature_seq(self, f: Feature) -> str:
        """Spliced feature sequence in its own orientation. Pieces are joined in GenBank (5'->3') order, so a
        gene that wraps around position 1 of a circular genome is assembled correctly. A mitochondrial CDS that
        ends in an incomplete stop codon (T or TA, completed by the poly-A tail) is completed with A."""
        s = "".join(self.seq[a:b] if f.strand == 1 else rc(self.seq[a:b]) for a, b in f.parts)
        if f.kind != "CDS":
            return s
        s = s[f.codon_start - 1:]
        rem = len(s) % 3
        if rem and f.table in (2, 3, 4, 5, 9, 13, 14, 16, 21, 22, 23, 24) and s.endswith("T" if rem == 1 else "TA"):
            s += "A" * (3 - rem)
        return s

    def summary(self) -> str:
        bits = [f"{len(self.seq):,} bases"]
        if self.kind == "annotated":
            n = len(self.cds)
            bits.append(f"{n} protein-coding gene{'s' if n != 1 else ''}")
            nc = sum(1 for f in self.features if f.kind in ("tRNA", "rRNA", "ncRNA"))
            if nc:
                bits.append(f"{nc} RNA genes")
        return ", ".join(bits)


# ---------------------------------------------------------------------------------------------
# Loading references
# ---------------------------------------------------------------------------------------------
def from_genbank(text: str) -> list[Reference]:
    from Bio import SeqIO
    out = []
    for rec in SeqIO.parse(io.StringIO(text), "genbank"):
        seq = str(rec.seq).upper().replace("U", "T")
        src = next((f for f in rec.features if f.type == "source"), None)
        q = src.qualifiers if src is not None else {}
        organism = rec.annotations.get("organism", "") or q.get("organism", [""])[0]
        organelle = q.get("organelle", [""])[0].lower()
        taxonomy = " ".join(rec.annotations.get("taxonomy", [])).lower()
        mito = "mitochondri" in organelle or "mitochondri" in rec.description.lower()
        euk_nuclear = ("eukaryota" in taxonomy and not mito and "chloroplast" not in organelle)
        feats = []
        for f in rec.features:
            if f.type not in ("CDS", "mRNA", "gene") + NONCODING_TYPES or f.type in ("misc_feature", "repeat_region"):
                continue
            fq = f.qualifiers
            name = (fq.get("gene") or fq.get("locus_tag") or fq.get("product") or fq.get("note") or [f.type])[0]
            parts = [(int(p.start), int(p.end)) for p in f.location.parts]
            strand = f.location.strand or 1
            feats.append(Feature(f.type, name, parts, strand, int(fq.get("transl_table", ["0"])[0]) or 0,
                                 fq.get("product", [""])[0], int(fq.get("codon_start", ["1"])[0])))
        default_table = next((x.table for x in feats if x.kind == "CDS" and x.table), 0) or (2 if mito else 1)
        for x in feats:
            x.table = x.table or default_table
        kind = "annotated" if any(x.kind == "CDS" for x in feats) else "genomic"
        out.append(Reference(rec.id, rec.description, seq, feats, kind, "m" if mito else "g", organism,
                             default_table, euk_nuclear))
    return out


def load(text: str, name: str = "") -> list[Reference]:
    """GenBank, FASTA or plain sequence text -> references. Raises M.BadNotation with a reason."""
    t = text.lstrip()
    if t.startswith("LOCUS"):
        refs = from_genbank(text)
        if not refs:
            raise M.BadNotation("The GenBank file has no sequence records.")
        return refs
    records = []
    if t.startswith(">"):
        for block in t.split(">")[1:]:
            head, _, body = block.partition("\n")
            records.append((head.strip(), body))
    else:
        records.append((name or "reference", t))
    out = []
    for head, body in records:
        p = S.parse(body)
        if p.kind not in ("DNA", "RNA"):
            raise M.BadNotation(f"'{head[:40]}' is not a DNA or RNA sequence. {p.reason}".strip())
        dna = S.to_dna(p.seq)
        rid = head.split()[0] if head else "reference"
        is_cds = (dna.startswith("ATG") and len(dna) % 3 == 0 and dna[-3:] in ("TAA", "TAG", "TGA")
                  and "*" not in S.translate(dna[:-3]))
        out.append(Reference(rid, head, dna, [], "coding" if is_cds else "genomic", "c" if is_cds else "g"))
    return out


def read_file(path: str | Path, name: str) -> str:
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


# ---------------------------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------------------------
@dataclass
class Variant:
    chrom: str
    pos: int          # 1-based, as in VCF
    ref: str
    alt: str
    id: str = ""

    def core(self) -> tuple[int, str, str]:
        """Trim shared prefix and suffix: (0-based start, ref core, alt core)."""
        r, a, p = self.ref, self.alt, self.pos - 1
        while r and a and r[0] == a[0]:
            r, a, p = r[1:], a[1:], p + 1
        while r and a and r[-1] == a[-1]:
            r, a = r[:-1], a[:-1]
        return p, r, a


_SIMPLE = re.compile(r"^(?:[gm]\.)?([ACGT])?(\d+)([ACGT])$|^(?:[gm]\.)?(\d+)([ACGT])>([ACGT])$", re.I)


def parse_genomic_change(text: str, ref: Reference) -> Variant:
    """Accepts A23403G, 23403A>G, g.23403A>G, m.3243A>G, g.21765_21770del, g.100_101insT, g.100dup."""
    t = text.strip().replace(" ", "")
    if ref.kind == "coding" and t.lower().startswith("c."):
        m = M.parse_hgvs(t, ref.seq)
        return mutation_to_variant(m, ref)
    if m := re.match(r"^([ACGT])(\d+)([ACGT])$", t, re.I):
        pos, r, a = int(m[2]), m[1].upper(), m[3].upper()
    elif m := re.match(r"^(?:[gmc]\.)?(\d+)([ACGT])>([ACGT])$", t, re.I):
        pos, r, a = int(m[1]), m[2].upper(), m[3].upper()
    else:
        mm = re.match(r"^(?:[gmc]\.)(.*)$", t, re.I)
        if not mm:
            raise M.BadNotation("Write the change like A23403G, 23403A>G, g.23403A>G, m.3243A>G, g.100_105del, "
                                "g.100_101insT or g.100_102dup.")
        m_ = M.parse_hgvs("c." + mm[1], ref.seq)      # same grammar, genomic coordinates
        return mutation_to_variant(m_, ref)
    if not 1 <= pos <= len(ref.seq):
        raise M.BadNotation(f"Position {pos} is outside the reference (1 to {len(ref.seq):,}).")
    if ref.seq[pos - 1] != r:
        raise M.BadNotation(f"Position {pos} is {ref.seq[pos - 1]} in the reference, not {r}.")
    return Variant(ref.id, pos, r, a)


def mutation_to_variant(m: M.Mutation, ref: Reference) -> Variant:
    s = ref.seq
    if m.kind == "substitution":
        return Variant(ref.id, m.start, m.ref, m.alt)
    if m.kind == "deletion":
        p = m.start - 1
        anchor = s[p - 1] if p >= 1 else s[m.end]
        return (Variant(ref.id, p, anchor + m.ref, anchor) if p >= 1 else
                Variant(ref.id, 1, m.ref + anchor, anchor))
    if m.kind in ("insertion", "duplication"):
        after = m.start if m.kind == "insertion" else m.end
        return Variant(ref.id, after, s[after - 1], s[after - 1] + m.alt)
    return Variant(ref.id, m.start, m.ref, m.alt)          # inversion, delins


def dna_class(v: Variant) -> dict:
    p, r, a = v.core()
    if len(r) == 1 and len(a) == 1:
        kind = "SNV"
        ti = (r, a) in TRANSITIONS
        rr, aa = (r, a) if r in "CT" else (r.translate(_COMP), a.translate(_COMP))
        return {"type": kind, "ti_tv": "Transition" if ti else "Transversion", "spectrum": f"{rr}>{aa}"}
    if len(r) == len(a):
        kind = "MNV"
    elif not r:
        kind = "Insertion"
    elif not a:
        kind = "Deletion"
    else:
        kind = "Delins"
    return {"type": kind, "ti_tv": "", "spectrum": ""}


def hgvs_genomic(v: Variant, ref: Reference) -> str:
    p, r, a = v.core()
    pre = ref.prefix if ref.prefix != "c" else "c"
    start = p + 1
    if len(r) == 1 and len(a) == 1:
        return f"{pre}.{start}{r}>{a}"
    m = None
    if not r:
        m = M.Mutation("insertion", p, p, "", a)        # insertion after base p (1-based)
    elif not a:
        m = M.Mutation("deletion", start, start + len(r) - 1, r, "")
    else:
        m = M.Mutation("delins", start, start + len(r) - 1, r, a)
    m = local_normalise(ref.seq, m)
    return pre + M.hgvs_of(m)[1:]


def local_normalise(seq: str, m: M.Mutation, flank: int = 300) -> M.Mutation:
    """HGVS normalisation (3' rule, dup) computed on a window around the change, so it scales to genomes."""
    lo = max(0, m.start - 1 - flank)
    hi = min(len(seq), m.end + flank)
    sub = seq[lo:hi]
    mm = M.Mutation(m.kind, m.start - lo, m.end - lo, m.ref, m.alt)
    try:
        n, _ = M.normalise(sub, mm)
    except Exception:
        return m
    return M.Mutation(n.kind, n.start + lo, n.end + lo, n.ref, n.alt)


def parse_vcf(text: str) -> tuple[list[Variant], list[str]]:
    variants, problems = [], []
    seen_header = False
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            seen_header = True
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 5:
            f = line.split()
        if len(f) < 5:
            problems.append(f"Line {n}: fewer than 5 columns.")
            continue
        chrom, pos, vid, ref, alts = f[0], f[1], f[2], f[3].upper(), f[4].upper()
        if not pos.isdigit():
            problems.append(f"Line {n}: position '{pos}' is not a number.")
            continue
        for alt in alts.split(","):
            if not re.fullmatch(r"[ACGTN]+", ref) or not re.fullmatch(r"[ACGTN]+", alt):
                problems.append(f"Line {n}: {ref}>{alt} is a symbolic or unsupported allele, skipped.")
                continue
            variants.append(Variant(chrom, int(pos), ref, alt, "" if vid == "." else vid))
    if not seen_header and not variants:
        problems.insert(0, "This does not look like a VCF file (no #CHROM header line).")
    return variants, problems


def looks_like_vcf(text: str) -> bool:
    head = text[:5000]
    return head.startswith("##fileformat=VCF") or "\n#CHROM" in head or head.startswith("#CHROM")


# ---------------------------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------------------------
def _mutation_in_feature(ref: Reference, f: Feature, v: Variant) -> tuple[M.Mutation | None, str]:
    """Translate a genomic variant into a c. mutation of feature f (1-based c. coordinates)."""
    p, r, a = v.core()
    if f.strand == -1:
        r, a = rc(r), rc(a)
    off = f.codon_start - 1
    if r:
        idx = [f.to_cds_index(p + i) for i in range(len(r))]
        if any(i is None for i in idx):
            return None, "The change crosses the edge of the coding sequence."
        lo, hi = min(idx) - off, max(idx) - off
        if hi - lo + 1 != len(r) or lo < 0:
            return None, "The change spans an intron or the edge of the coding sequence."
        if not a:
            return M.Mutation("deletion", lo + 1, hi + 1, r, ""), ""
        if len(r) == 1 and len(a) == 1:
            return M.Mutation("substitution", lo + 1, lo + 1, r, a), ""
        return M.Mutation("delins", lo + 1, hi + 1, r, a), ""
    # insertion between genomic p-1 and p
    i1, i2 = f.to_cds_index(p - 1), f.to_cds_index(p)
    if i1 is None or i2 is None or abs(i1 - i2) != 1:
        return None, "The insertion is at the edge of the coding sequence."
    before = min(i1, i2) - off
    return M.Mutation("insertion", before + 1, before + 1, "", a), ""


def _short_protein(name: str, prot_change: str) -> str:
    """S:D614G style, as used in virology and Nextclade."""
    one = {v[0]: k for k, v in S.AMINO_ACIDS.items()}
    one["Ter"] = "*"
    m = re.match(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$", prot_change)
    if m and m[1] in one and m[3] in one:
        return f"{name}:{one[m[1]]}{m[2]}{one[m[3]]}"
    m = re.match(r"^p\.([A-Z][a-z]{2})(\d+)=$", prot_change)
    if m and m[1] in one:
        return f"{name}:{one[m[1]]}{m[2]}{one[m[1]]}"
    return f"{name}:{prot_change[2:]}" if prot_change.startswith("p.") else ""


def annotate(ref: Reference, v: Variant, splice: bool = False) -> dict:
    """Everything known about one variant: DNA class, location, gene effects and ML splice impact."""
    p, r, a = v.core()
    row = {"chrom": v.chrom, "pos": v.pos, "ref": v.ref, "alt": v.alt, "id": v.id, **dna_class(v),
           "hgvs_g": "", "region": "", "gene": "", "hgvs_c": "", "hgvs_p": "", "protein_short": "",
           "effect": "", "severity": "", "codon": "", "splice_max_delta": None, "splice_note": "", "status": ""}
    if not r and not a:
        row.update(effect="No change", severity="neutral", status="REF and ALT are identical")
        return row
    if v.pos < 1 or p + len(r) > len(ref.seq):
        row["status"] = f"Position outside the reference (1 to {len(ref.seq):,})."
        return row
    if ref.seq[v.pos - 1: v.pos - 1 + len(v.ref)] != v.ref:
        row["status"] = (f"REF {v.ref} does not match the reference ({ref.seq[v.pos - 1: v.pos - 1 + len(v.ref)]}) "
                         f"at position {v.pos}.")
        return row
    row["hgvs_g"] = hgvs_genomic(v, ref)
    effects = []
    if ref.kind == "coding":
        effects.append(_effect_for(ref, Feature("CDS", ref.id, [(0, len(ref.seq))], 1, ref.table), v))
    cds_hit = False
    for f in ref.features:
        if f.kind in ("gene", "mRNA"):
            continue
        hit = (any(a_ < p + len(r) and p < b_ for a_, b_ in f.parts) if r else
               (f.contains(p - 1) and f.contains(p)))
        if not hit:
            continue
        if f.kind == "CDS":
            cds_hit = True
            effects.append(_effect_for(ref, f, v))
        else:
            effects.append({"region": f.kind, "gene": "" if f.name == f.kind else f.name,
                            "effect": f"Non-coding RNA gene ({f.kind})"
                            if f.kind in ("tRNA", "rRNA", "ncRNA", "misc_RNA") else f"Non-coding region ({f.kind})",
                            "severity": "unknown"})
    if not cds_hit:
        effects += _transcript_context(ref, p, max(1, len(r)), r, a)
    elif len(r) > 1 and any(e.get("effect", "").startswith("Complex") for e in effects):
        # a deletion that runs from an exon into an intron: does it remove a splice site?
        for i in range(len(r)):
            for c in _transcript_context(ref, p + i, 1):
                if c["effect"] in ("Splice donor site", "Splice acceptor site"):
                    effects.append({**c, "hgvs_c": "", "status": "Annotated. The deletion removes part of the exon "
                                                                  "and the canonical splice site next to it."})
                    break
            else:
                continue
            break
    if not effects:
        effects.append({"region": "intergenic" if ref.kind == "annotated" else "genomic", "gene": "",
                        "effect": "Intergenic" if ref.kind == "annotated" else "Not annotated",
                        "severity": "unknown"})
    order = {"severe": 0, "moderate": 1, "unknown": 2, "neutral": 3}
    effects.sort(key=lambda e: order.get(e.get("severity", "unknown"), 2))
    top = effects[0]
    row.update({k: top.get(k, "") for k in ("region", "gene", "hgvs_c", "hgvs_p", "protein_short", "effect",
                                            "severity", "codon")})
    if len(effects) > 1:
        row["also"] = "; ".join(f"{e.get('gene')}: {e.get('effect')} {e.get('hgvs_p', '')}".strip()
                                for e in effects[1:])
    row["status"] = "Annotated" if not top.get("status") else top["status"]
    if splice:
        row.update(splice_scores(ref, v))
    return row


def _transcript_context(ref: Reference, p: int, n: int, r: str = "", alt: str = "") -> list[dict]:
    """Intron, splice site and UTR labels for variants outside a CDS but inside a spliced gene."""
    out = []
    for f in ref.cds:
        if len(f.parts) < 2 and not any(m.kind == "mRNA" and m.name == f.name for m in ref.features):
            continue
        tx = next((m for m in ref.features if m.kind == "mRNA" and m.name == f.name), None)
        lo, hi = (tx.start, tx.end) if tx else (f.start, f.end)
        if not (lo <= p < hi):
            continue
        exons = sorted(tx.parts) if tx else sorted(f.parts)
        in_exon = any(a <= p < b for a, b in exons)
        if in_exon:
            before = (p < f.start) if f.strand == 1 else (p >= f.end)
            label = "5' UTR" if before else "3' UTR"
            out.append({"region": label, "gene": f.name, "effect": label, "severity": "unknown"})
            continue
        # intronic: distance to the nearest exon edge, in transcript orientation
        best = None
        for a, b in exons:
            for edge, side in ((b, "after"), (a, "before")):
                d = (p - edge + 1) if side == "after" else (edge - p)
                if d >= 1 and (best is None or d < best[0]):
                    best = (d, side, a, b)
        d, side, a, b = best
        if f.strand == 1:
            donor = side == "after"          # just after an exon end: the 5' end of the intron
        else:
            donor = side == "before"
        hg = ""
        cpos_edge = (b - 1) if side == "after" else a
        k = f.to_cds_index(cpos_edge)
        if k is not None:
            k = k - (f.codon_start - 1) + 1
            hg = f"c.{k}+{d}" if donor else f"c.{k}-{d}"
            if len(r) == 1 and len(alt) == 1:
                rr, aa = (r, alt) if f.strand == 1 else (rc(r), rc(alt))
                hg += f"{rr}>{aa}"
        if d <= 2:
            label, sev = ("Splice donor site" if donor else "Splice acceptor site"), "severe"
        elif d <= 8:
            label, sev = "Splice region", "moderate"
        else:
            label, sev = "Intronic", "unknown"
        site = f"{'+' if donor else '-'}{d}"
        out.append({"region": "intron", "gene": f.name, "effect": label, "severity": sev,
                    "hgvs_c": hg + ("" if n == 1 else f" (first base, {n} bases changed)"),
                    "status": f"Annotated. Position {site} from the nearest exon boundary."})
    return out


def _effect_for(ref: Reference, f: Feature, v: Variant) -> dict:
    m, why = _mutation_in_feature(ref, f, v)
    out = {"region": "CDS", "gene": f.name}
    if m is None:
        out.update(effect="Complex (crosses a coding boundary)", severity="moderate", status=why)
        return out
    cds = ref.feature_seq(f)
    m = local_normalise(cds, m)
    m.hgvs = M.hgvs_of(m)
    mut = M.apply(cds, m)
    e = M.effect(cds, mut, m, f.table)
    out.update(hgvs_c=m.hgvs, hgvs_p=e.protein_change, effect=e.label, severity=e.severity,
               protein_short=_short_protein(f.name, e.protein_change),
               codon=f"{e.ref_codon}>{e.mut_codon}" if e.ref_codon else "")
    return out


def splice_scores(ref: Reference, v: Variant, strands: tuple[int, ...] | None = None) -> dict:
    try:
        return _splice_scores(ref, v, strands)
    except Exception as e:  # noqa: BLE001  a model problem must never break variant annotation
        return {"splice_max_delta": None, "splice_note": f"Splice prediction unavailable ({type(e).__name__})."}


def _splice_scores(ref: Reference, v: Variant, strands: tuple[int, ...] | None = None) -> dict:
    """ML splice impact. Uses the gene's strand when the reference is annotated, otherwise both strands
    and reports the larger change. Profile positions are 1-based reference coordinates."""
    from . import ml
    if not ml.available():
        return {"splice_max_delta": None, "splice_note": "Splice models not installed."}
    p, r, a = v.core()
    lo, hi = max(0, p - 300), min(len(ref.seq), p + len(r) + 300)
    rs = ref.seq[lo:hi]
    ms = rs[:p - lo] + a + rs[p - lo + len(r):]
    if strands is None:
        g = gene_strand(ref, v)
        strands = (g,) if g else (1, -1)
    best = None
    for strand in strands:
        if strand == 1:
            x = ml.splice_impact(rs, ms, p - lo, p - lo + len(r), p - lo + len(a))
            conv = (lambda q: None if q is None else int(round(lo + q + 1)))
            to_ref = {k: {lo + q + 1: sc for q, sc in d.items()} for k, d in x["profile"].items()}
        else:
            R, Mt = rc(rs), rc(ms)
            start = len(rs) - (p - lo + len(r))
            x = ml.splice_impact(R, Mt, start, start + len(r), start + len(a))
            conv = (lambda q, n=len(rs): None if q is None else int(round(lo + (n - 1 - q) + 1)))
            to_ref = {k: {lo + (len(rs) - 1 - q) + 1: sc for q, sc in d.items()} for k, d in x["profile"].items()}
        x["profile"] = to_ref
        for k in ("donor", "acceptor"):
            x[k]["gain_pos"], x[k]["loss_pos"] = conv(x[k]["gain_pos"]), conv(x[k]["loss_pos"])
        x["strand"] = "+" if strand == 1 else "-"
        if best is None or x["max_delta"] > best["max_delta"]:
            best = x
    kinds = []
    for k in ("donor", "acceptor"):
        if best[k]["gain"] >= 0.2:
            kinds.append(f"{k} gain {best[k]['gain']:+.2f}")
        if best[k]["loss"] <= -0.2:
            kinds.append(f"{k} loss {best[k]['loss']:+.2f}")
    note = (", ".join(kinds) + f" ({best['strand']} strand)") if kinds else "No predicted splice change (all |Δ| < 0.20)"
    return {"splice_max_delta": round(best["max_delta"], 3), "splice_note": note, "splice_detail": best}


def summarise(rows: list[dict]) -> dict:
    snv = [r for r in rows if r.get("type") == "SNV" and r.get("status", "").startswith("Annotated")]
    ti = sum(1 for r in snv if r["ti_tv"] == "Transition")
    tv = len(snv) - ti
    spectrum = {k: sum(1 for r in snv if r["spectrum"] == k) for k in SPECTRUM}
    genes: dict[str, int] = {}
    for r in rows:
        if r.get("gene"):
            genes[r["gene"]] = genes.get(r["gene"], 0) + 1
    return {"n": len(rows), "annotated": sum(1 for r in rows if r.get("status", "").startswith("Annotated")),
            "snv": len(snv), "ti": ti, "tv": tv, "ti_tv": (ti / tv) if tv else None, "spectrum": spectrum,
            "genes": dict(sorted(genes.items(), key=lambda kv: -kv[1]))}


def _lis(vals: list[int]) -> list[int]:
    """Indices of a longest strictly increasing subsequence (patience sorting)."""
    import bisect
    tails, tails_i, prev = [], [], [-1] * len(vals)
    for i, v in enumerate(vals):
        k = bisect.bisect_left(tails, v)
        if k == len(tails):
            tails.append(v)
            tails_i.append(i)
        else:
            tails[k], tails_i[k] = v, i
        prev[i] = tails_i[k - 1] if k else -1
    out, i = [], tails_i[-1] if tails_i else -1
    while i != -1:
        out.append(i)
        i = prev[i]
    return out[::-1]


def compare_genomes(ref: Reference, query: str, max_len: int = 1_000_000, k: int = 24,
                    stride: int = 4) -> list[Variant]:
    """Variants of a closely related query genome (e.g. a consensus sequence) against the reference.

    Exact k-mer anchors are chained in order, and only the stretches between anchors are aligned, so
    a 30 kb viral genome takes well under a second. Leading and trailing overhangs and stretches of N
    in the query are not reported as variants."""
    R = ref.seq
    Q = S.to_dna("".join(ch for ch in query.upper() if ch.isalpha()))
    if len(Q) > max_len or len(R) > max_len:
        raise M.BadNotation(f"Whole-genome comparison is limited to {max_len:,} bases.")
    index: dict[str, int] = {}
    for i in range(0, len(R) - k + 1, stride):
        km = R[i:i + k]
        index[km] = -1 if km in index else i
    anchors = []
    for j in range(len(Q) - k + 1):
        i = index.get(Q[j:j + k])
        if i is not None and i >= 0:
            anchors.append((j, i))
    if not anchors:
        raise M.BadNotation("The query shares no exact stretches with the reference, so it is not the same genome.")
    keep = _lis([r for _, r in anchors])
    chain = [anchors[i] for i in keep]
    blocks = []                                          # (q0, r0, length) exact-match blocks
    for q0, r0 in chain:
        if blocks and q0 - r0 == blocks[-1][0] - blocks[-1][1] and q0 <= blocks[-1][0] + blocks[-1][2]:
            bq, br, bl = blocks[-1]
            blocks[-1] = (bq, br, max(bl, q0 + k - bq))
        else:
            if blocks and (q0 < blocks[-1][0] + blocks[-1][2] or r0 < blocks[-1][1] + blocks[-1][2]):
                continue
            blocks.append((q0, r0, k))
    out: list[Variant] = []
    # ends: align what lies before the first and after the last anchor, keeping substitutions and any
    # internal indel, but not indels that only reflect a shorter or longer query end
    (fq, fr, _), (lq, lr, ll) = blocks[0], blocks[-1]
    for rseg, qseg, lo_r, edge in ((R[:fr + 8], Q[:fq + 8], 0, "start"),
                                   (R[lr + ll - 8:], Q[lq + ll - 8:], lr + ll - 8, "end")):
        if not rseg or not qseg or max(len(rseg), len(qseg)) > 20000 or rseg == qseg:
            continue
        if "N" in qseg and len(qseg) == len(rseg):
            qseg = "".join(x if c == "N" else c for x, c in zip(rseg, qseg))
        for m in M.detect(rseg, qseg):
            if "N" in m.alt:
                continue
            touches = (m.start <= 1) if edge == "start" else (m.end >= len(rseg))
            if m.kind != "substitution" and touches:
                continue
            if m.kind != "substitution" and len(rseg) != len(qseg):
                continue
            shifted = M.Mutation(m.kind, m.start + lo_r, m.end + lo_r, m.ref, m.alt)
            if shifted.kind == "duplication":
                shifted.ref = R[shifted.start - 1:shifted.end]
            out.append(mutation_to_variant(shifted, ref))
    for (q1, r1, l1), (q2, r2, _) in zip(blocks, blocks[1:]):
        qe, re_ = q1 + l1, r1 + l1
        if qe == q2 and re_ == r2:
            continue
        lo_r, hi_r = re_ - 8, r2 + 8
        lo_q, hi_q = qe - 8, q2 + 8
        rseg, qseg = R[lo_r:hi_r], Q[lo_q:hi_q]
        if max(len(rseg), len(qseg)) > 20000:
            continue
        if "N" in qseg:                                  # unsequenced bases are not variants
            if len(qseg) != len(rseg):
                continue
            qseg = "".join(r if c == "N" else c for r, c in zip(rseg, qseg))
        for m in M.detect(rseg, qseg):
            alt_seg = m.alt
            if "N" in alt_seg:
                continue
            shifted = M.Mutation(m.kind, m.start + lo_r, m.end + lo_r, m.ref, m.alt)
            if shifted.kind == "duplication":
                shifted.ref = R[shifted.start - 1:shifted.end]
            out.append(mutation_to_variant(shifted, ref))
    out.sort(key=lambda v: v.pos)
    return out


def cds_effect(ref: Reference, v: Variant):
    """(feature, cds, mutant cds, mutation, Effect) for the first coding sequence the variant changes."""
    feats = ([Feature("CDS", ref.id, [(0, len(ref.seq))], 1, ref.table)] if ref.kind == "coding" else ref.cds)
    p, r, a = v.core()
    for f in feats:
        hit = (any(a_ < p + len(r) and p < b_ for a_, b_ in f.parts) if r else (f.contains(p - 1) and f.contains(p)))
        if not hit:
            continue
        m, _ = _mutation_in_feature(ref, f, v)
        if m is None:
            continue
        cds = ref.feature_seq(f)
        m = local_normalise(cds, m)
        m.hgvs = M.hgvs_of(m)
        mut = M.apply(cds, m)
        return f, cds, mut, m, M.effect(cds, mut, m, f.table)
    return None


def gene_strand(ref: Reference, v: Variant) -> int | None:
    p = v.pos - 1
    for f in ref.features:
        if f.kind in ("CDS", "mRNA", "gene") and f.start - 60 <= p < f.end + 60:
            return f.strand
    return 1 if ref.kind == "coding" else None
