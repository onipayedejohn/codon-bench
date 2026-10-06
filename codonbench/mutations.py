"""Mutations: apply, detect and classify changes to a coding sequence.

Coordinates follow HGVS coding (c.) numbering: c.1 is the A of the ATG start
codon, and protein (p.) numbering counts the initiator methionine as residue 1.
Older haemoglobin literature does not count the methionine, so HbS is "Glu6Val"
there and p.Glu7Val in HGVS. The app shows both for HBB.

Point and small mutations: substitution (silent, missense, nonsense, stop-loss,
start-loss), insertion, deletion, duplication (in-frame or frameshift).
Structural mutations: inversion and translocation, shown on a chromosome-style
diagram and checked for their effect on the protein.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from Bio import Align

from . import seq as S

THREE = {k: v[0] for k, v in S.AMINO_ACIDS.items()}
THREE["*"] = "Ter"

EFFECT_INFO = {
    "Silent": ("The codon changes but still codes for the same amino acid (or a stop codon stays a stop), because "
               "the genetic code is degenerate. The protein is unchanged.", "neutral"),
    "Missense": ("One amino acid is replaced by another. The effect ranges from none to severe, depending on where "
                 "it is and how different the new amino acid is.", "moderate"),
    "Nonsense": ("A codon becomes a stop codon, so translation ends early and the protein is cut short. The mRNA is "
                 "often destroyed by nonsense-mediated decay.", "severe"),
    "Stop-loss": ("The stop codon is lost, so the ribosome reads on into the 3' untranslated region and the protein "
                  "gets longer.", "severe"),
    "Start-loss": ("The start codon is lost, so translation cannot begin at the normal place.", "severe"),
    "Frameshift": ("The number of bases added or removed is not a multiple of three, so every codon after it is read "
                   "in the wrong frame until a new stop codon appears.", "severe"),
    "In-frame insertion": ("Whole codons are added, so amino acids are inserted but the reading frame is kept.",
                           "moderate"),
    "In-frame deletion": ("Whole codons are removed, so amino acids are lost but the reading frame is kept.",
                          "moderate"),
    "Duplication": ("A stretch of bases is copied and inserted right after the original.", "moderate"),
    "Inversion": ("A segment is cut out, turned around and put back, so it reads as its reverse complement.",
                  "moderate"),
    "Translocation": ("Two non-homologous chromosomes break and swap the pieces. No DNA is lost in a balanced "
                      "translocation, but a gene cut at a breakpoint can be broken, or joined to part of another gene "
                      "to make a fusion gene.", "severe"),
    "No change": ("The two sequences are identical.", "neutral"),
    "Splice donor site": ("The change hits the GT at the start of an intron (+1 or +2), which the spliceosome needs "
                          "to remove the intron. Exon skipping or intron retention is likely.", "severe"),
    "Splice acceptor site": ("The change hits the AG at the end of an intron (-1 or -2), which the spliceosome needs "
                             "to join the next exon. Exon skipping is likely.", "severe"),
    "Splice region": ("The change is within a few bases of an exon boundary, where it can weaken the splice "
                      "site. Check the ML splice score.", "moderate"),
    "Intronic": ("The change is inside an intron, away from the splice sites. Most such changes have no effect, but "
                 "some create a new splice site.", "unknown"),
    "5' UTR": ("The change is in the 5' untranslated region, before the start codon. It does not change the "
               "protein but can affect translation.", "unknown"),
    "3' UTR": ("The change is in the 3' untranslated region, after the stop codon. It does not change the protein "
               "but can affect mRNA stability.", "unknown"),
    "Intergenic": ("The change is outside every annotated gene.", "unknown"),
    "Not annotated": ("The reference has no gene annotation, so only the DNA change is described.", "unknown"),
    "Complex (crosses a coding boundary)": ("The change spans the edge of a coding sequence or an intron, so its "
                                            "effect cannot be worked out codon by codon.", "moderate"),
}


@dataclass
class Mutation:
    kind: str                      # substitution, insertion, deletion, duplication, inversion, delins
    start: int                     # 1-based c. position (for insertion: the base before the insertion)
    end: int                       # inclusive (= start for substitution and insertion)
    ref: str = ""
    alt: str = ""
    hgvs: str = ""


@dataclass
class Effect:
    label: str                     # Silent, Missense, Nonsense, Frameshift, ...
    severity: str                  # neutral, moderate, severe
    explanation: str
    dna_change: str                # HGVS c.
    protein_change: str            # HGVS p.
    ref_protein: str
    mut_protein: str
    first_diff: int | None         # 1-based residue where proteins first differ
    ref_codon: str = ""
    mut_codon: str = ""
    ref_aa: str = ""
    mut_aa: str = ""
    class_change: str = ""
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# HGVS-style input
# ---------------------------------------------------------------------------------------------
_SUB = re.compile(r"^c\.(\d+)([ACGT])>([ACGT])$", re.I)
_DEL = re.compile(r"^c\.(\d+)(?:_(\d+))?del([ACGT]*)$", re.I)
_INS = re.compile(r"^c\.(\d+)_(\d+)ins([ACGT]+)$", re.I)
_DUP = re.compile(r"^c\.(\d+)(?:_(\d+))?dup([ACGT]*)$", re.I)
_INV = re.compile(r"^c\.(\d+)_(\d+)inv$", re.I)
_DELINS = re.compile(r"^c\.(\d+)(?:_(\d+))?delins([ACGT]+)$", re.I)


class BadNotation(ValueError):
    pass


def parse_hgvs(text: str, cds: str) -> Mutation:
    t = text.strip().replace(" ", "")
    n = len(cds)

    def chk(*pos):
        for p in pos:
            if p < 1 or p > n:
                raise BadNotation(f"Position {p} is outside the sequence (1 to {n}).")

    if m := _SUB.match(t):
        p, r, a = int(m[1]), m[2].upper(), m[3].upper()
        chk(p)
        if cds[p - 1] != r:
            raise BadNotation(f"c.{p} is {cds[p - 1]} in this sequence, not {r}.")
        return Mutation("substitution", p, p, r, a)
    if m := _DELINS.match(t):
        a, b = int(m[1]), int(m[2] or m[1])
        chk(a, b)
        return Mutation("delins", a, b, cds[a - 1:b], m[3].upper())
    if m := _DEL.match(t):
        a, b = int(m[1]), int(m[2] or m[1])
        chk(a, b)
        if m[3] and m[3].upper() != cds[a - 1:b]:
            raise BadNotation(f"c.{a}_{b} is {cds[a - 1:b]} in this sequence, not {m[3].upper()}.")
        return Mutation("deletion", a, b, cds[a - 1:b], "")
    if m := _INS.match(t):
        a, b = int(m[1]), int(m[2])
        chk(a, b)
        if b != a + 1:
            raise BadNotation("An insertion is written between two neighbouring positions, e.g. c.20_21insT.")
        return Mutation("insertion", a, a, "", m[3].upper())
    if m := _DUP.match(t):
        a, b = int(m[1]), int(m[2] or m[1])
        chk(a, b)
        return Mutation("duplication", a, b, cds[a - 1:b], cds[a - 1:b])
    if m := _INV.match(t):
        a, b = int(m[1]), int(m[2])
        chk(a, b)
        if b <= a:
            raise BadNotation("An inversion needs at least two bases, e.g. c.40_45inv.")
        return Mutation("inversion", a, b, cds[a - 1:b], S.reverse_complement(cds[a - 1:b]))
    raise BadNotation("Use HGVS coding notation, for example c.20A>T, c.118C>T, c.126_129del, c.20_21insT, "
                      "c.10_15dup or c.40_45inv.")


def hgvs_of(m: Mutation) -> str:
    rng = f"{m.start}" if m.start == m.end else f"{m.start}_{m.end}"
    if m.kind == "substitution":
        return f"c.{m.start}{m.ref}>{m.alt}"
    if m.kind == "deletion":
        return f"c.{rng}del"
    if m.kind == "insertion":
        return f"c.{m.start}_{m.start + 1}ins{m.alt}"
    if m.kind == "duplication":
        return f"c.{rng}dup"
    if m.kind == "inversion":
        return f"c.{rng}inv"
    return f"c.{rng}delins{m.alt}"


def apply(cds: str, m: Mutation) -> str:
    a, b = m.start, m.end
    if m.kind == "substitution":
        return cds[:a - 1] + m.alt + cds[a:]
    if m.kind == "deletion":
        return cds[:a - 1] + cds[b:]
    if m.kind == "insertion":
        return cds[:a] + m.alt + cds[a:]
    if m.kind == "duplication":
        return cds[:b] + cds[a - 1:b] + cds[b:]
    if m.kind in ("inversion", "delins"):
        return cds[:a - 1] + m.alt + cds[b:]
    raise ValueError(m.kind)


# ---------------------------------------------------------------------------------------------
# Effect on the protein
# ---------------------------------------------------------------------------------------------
def _protein(cds: str, table_id: int, readthrough: str = "") -> str:
    """Translate from c.1 to the first stop codon (kept as '*'); read into readthrough if no stop."""
    s = S.to_dna(cds + readthrough)
    out = []
    tab = S.codon_table(table_id)
    for i in range(0, len(s) - 2, 3):
        cod = s[i:i + 3]
        aa = "*" if cod in tab.stop_codons else tab.forward_table.get(cod, "X")
        if i == 0 and table_id != 1 and cod in tab.start_codons:
            aa = "M"                                   # alternative initiation codons are read by initiator tRNA
        out.append(aa)
        if aa == "*":
            break
    return "".join(out)


def _p(aa: str) -> str:
    return THREE.get(aa, "Xaa")


def _residues(seg: str) -> str:
    return "".join(_p(a) for a in seg)


def _describe_protein(rp: str, mp: str, lost: int = 0) -> tuple[str, str, str]:
    """HGVS p. description of an in-frame change. Returns (label, p. string, note).

    Both proteins run to their first stop ('*'). The changed block is found by trimming the longest
    common start and then the longest common end, which places insertions and deletions as far
    towards the C-terminus as possible (the HGVS 3' rule)."""
    if rp == mp:
        return "Silent", "p.(=)", ""
    i = 0
    while i < min(len(rp), len(mp)) and rp[i] == mp[i]:
        i += 1
    if i == 0 and rp[:1] == "M":
        return "Start-loss", "p.Met1?", "Translation cannot start at the normal AUG, so the protein is unknown."
    if i < len(rp) and rp[i] == "*":
        ext = len(mp) - len(rp)
        return ("Stop-loss", f"p.Ter{i + 1}{_p(mp[i]) if i < len(mp) else 'Xaa'}extTer"
                f"{ext if mp.endswith('*') else '?'}", "")
    if i < len(mp) and mp[i] == "*" or (mp.endswith("*") and len(mp) < len(rp) - lost):
        # a stop codon appears early: describe up to the new stop
        e = len(mp) - 1                                  # index of the new stop in mp
        if e == i:
            prot = f"p.{_p(rp[i])}{i + 1}Ter"
        else:
            last = min(e, len(rp) - 1) - 1
            rng = f"{_p(rp[i])}{i + 1}" + (f"_{_p(rp[last])}{last + 1}" if last > i else "")
            prot = f"p.{rng}delins{_residues(mp[i:e + 1])}"
        return "Nonsense", prot, f"A stop codon appears at residue {len(mp)}, so the protein is {len(rp) - len(mp)} residues shorter."
    j = 0
    while j < min(len(rp), len(mp)) - i and rp[len(rp) - 1 - j] == mp[len(mp) - 1 - j]:
        j += 1
    rs, ms = rp[i:len(rp) - j], mp[i:len(mp) - j]
    if not rs and ms:
        k = len(ms)
        if i - k >= 0 and rp[i - k:i] == ms:
            d0, d1 = i - k + 1, i
            return "Insertion", f"p.{_p(rp[d0 - 1])}{d0}" + ("dup" if k == 1 else f"_{_p(rp[d1 - 1])}{d1}dup"), ""
        return "Insertion", f"p.{_p(rp[i - 1])}{i}_{_p(rp[i])}{i + 1}ins{_residues(ms)}", ""
    if rs and not ms:
        e = i + len(rs)
        return "Deletion", (f"p.{_p(rs[0])}{i + 1}del" if len(rs) == 1 else
                            f"p.{_p(rs[0])}{i + 1}_{_p(rs[-1])}{e}del"), ""
    if len(rs) == 1 and len(ms) == 1:
        return "Missense", f"p.{_p(rs)}{i + 1}{_p(ms)}", ""
    e = i + len(rs)
    rng = f"{_p(rs[0])}{i + 1}" + (f"_{_p(rs[-1])}{e}" if len(rs) > 1 else "")
    note = f"{len(rs)} residue{'s' if len(rs) != 1 else ''} replaced by {len(ms)}."
    return "Delins", f"p.{rng}delins{_residues(ms)}", note


def effect(ref_cds: str, mut_cds: str, m: Mutation | None = None, table_id: int = 1,
           utr3: str = "") -> Effect:
    """Classify the consequence of a change to a coding sequence that starts at c.1."""
    ref_cds, mut_cds = S.to_dna(ref_cds), S.to_dna(mut_cds)
    rp = _protein(ref_cds, table_id, utr3)
    mp = _protein(mut_cds, table_id, utr3)
    dna = hgvs_of(m) if m else ""
    first = next((i + 1 for i, (a, b) in enumerate(zip(rp, mp)) if a != b), None)
    if first is None and len(rp) != len(mp):
        first = min(len(rp), len(mp)) + 1
    notes = []
    if ref_cds == mut_cds:
        return Effect("No change", "neutral", EFFECT_INFO["No change"][0], dna, "p.(=)", rp, mp, None)

    kind = m.kind if m else None
    length_change = len(mut_cds) - len(ref_cds)
    rc = mc = ra = ma = cls = ""
    if kind == "substitution":
        ci = (m.start - 1) // 3
        rc, mc = ref_cds[ci * 3:ci * 3 + 3], mut_cds[ci * 3:ci * 3 + 3]
        ra, ma = (rp[ci] if ci < len(rp) else "?"), (mp[ci] if ci < len(mp) else "?")
        pos = ci + 1
        starts = {"ATG"} if table_id == 1 else set(S.codon_table(table_id).start_codons)
        if ci == 0 and rc in starts and mc not in starts:
            label, prot = "Start-loss", "p.Met1?"
        elif ra == ma:
            label, prot = "Silent", f"p.{_p(ra)}{pos}="
        elif ma == "*":
            label, prot = "Nonsense", f"p.{_p(ra)}{pos}Ter"
        elif ra == "*":
            ext = len(mp) - len(rp)
            label = "Stop-loss"
            prot = f"p.Ter{pos}{_p(ma)}extTer{ext if mp.endswith('*') else '?'}"
        else:
            label, prot = "Missense", f"p.{_p(ra)}{pos}{_p(ma)}"
            c1, c2 = S.AMINO_ACIDS[ra][2], S.AMINO_ACIDS[ma][2]
            cls = (f"{S.AMINO_ACIDS[ra][1]} ({c1.lower()}) to {S.AMINO_ACIDS[ma][1]} ({c2.lower()}): "
                   + ("same chemical class, a conservative change." if c1 == c2 else
                      "a different chemical class, a radical change that is more likely to disrupt the protein."))
        return Effect(label, EFFECT_INFO[label][1], EFFECT_INFO[label][0], dna, prot, rp, mp, first, rc, mc, ra, ma,
                      cls, notes)

    if length_change % 3 == 0:
        k = abs(length_change) // 3
        plabel, prot, pnote = _describe_protein(rp, mp, k if length_change < 0 else 0)
        if plabel == "Silent" and kind == "inversion":
            notes.append("The inverted bases code for the same amino acids, so the protein is unchanged.")
        if kind == "inversion":
            label = "Inversion"
        elif kind == "duplication":
            label = "Duplication"
            notes.append(f"The duplicated {m.end - m.start + 1} bases keep the frame, so {k} amino acid"
                         f"{'s are' if k != 1 else ' is'} added.")
        elif kind in ("insertion", "deletion") and plabel not in ("Nonsense", "Start-loss", "Stop-loss"):
            label = "In-frame insertion" if length_change > 0 else "In-frame deletion"
        elif plabel in ("Insertion", "Deletion", "Delins"):
            label = ("In-frame insertion" if length_change > 0 else "In-frame deletion" if length_change < 0
                     else "Missense")
        else:
            label = plabel
        if pnote:
            notes.append(pnote)
        return Effect(label, EFFECT_INFO[label][1], EFFECT_INFO[label][0], dna, prot, rp, mp, first, notes=notes)

    # frameshifts
    label = "Frameshift"
    if first is not None and first <= len(mp):
        new_stop = len(mp) - first + 1 if mp.endswith("*") else None
        prot = (f"p.{_p(rp[first - 1])}{first}{_p(mp[first - 1])}fsTer{new_stop}" if new_stop and mp[first - 1] != "*"
                else f"p.{_p(rp[first - 1])}{first}Ter" if mp[first - 1] == "*" else
                f"p.{_p(rp[first - 1])}{first}fs (no new stop codon in the sequence)")
    else:
        prot = "p.?"
    if mp.endswith("*"):
        notes.append(f"The new reading frame hits a stop codon after {len(mp) - 1} residues "
                     f"(normal protein: {len(rp) - 1}).")
    else:
        notes.append("No stop codon is reached in the new frame before the sequence ends.")
    if kind == "duplication" and length_change % 3:
        notes.insert(0, f"The duplicated {m.end - m.start + 1} bases are not a multiple of three, so this "
                        "duplication also shifts the frame.")
    return Effect(label, EFFECT_INFO[label][1], EFFECT_INFO[label][0], dna, prot, rp, mp, first, notes=notes)


# ---------------------------------------------------------------------------------------------
# Comparing two sequences
# ---------------------------------------------------------------------------------------------
def _aligner() -> Align.PairwiseAligner:
    a = Align.PairwiseAligner()
    a.mode = "global"
    a.match_score, a.mismatch_score = 2, -3
    a.open_gap_score, a.extend_gap_score = -6, -1
    a.end_gap_score = 0
    return a


def detect(ref: str, mut: str, max_len: int = 20000) -> list[Mutation]:
    """List the differences between a reference and a mutant sequence, in reference coordinates."""
    ref, mut = S.to_dna(ref), S.to_dna(mut)
    if ref == mut:
        return []
    if len(ref) > max_len or len(mut) > max_len:
        raise BadNotation(f"Comparison is limited to {max_len:,} bases per sequence.")
    if len(ref) == len(mut):
        inv = _find_inversion(ref, mut)
        if inv:
            return [inv]
    aln = _aligner().align(ref, mut)[0]
    a_ref, a_mut = _gapped(aln)
    out: list[Mutation] = []
    i = 0
    rpos = 0           # bases of ref consumed
    n = len(a_ref)
    while i < n:
        r, q = a_ref[i], a_mut[i]
        if r == q:
            rpos += 1
            i += 1
            continue
        j = i
        while j < n and a_ref[j] != a_mut[j]:
            j += 1
        seg_r = a_ref[i:j].replace("-", "")
        seg_m = a_mut[i:j].replace("-", "")
        start = rpos + 1
        if seg_r and not seg_m:
            a0, b0 = start, start + len(seg_r) - 1
            while b0 < len(ref) and ref[a0 - 1] == ref[b0]:          # HGVS 3' rule: shift right
                a0, b0 = a0 + 1, b0 + 1
            m = Mutation("deletion", a0, b0, ref[a0 - 1:b0], "")
        elif seg_m and not seg_r:
            pos, ins = rpos, seg_m
            while pos < len(ref) and ins[0] == ref[pos]:                # HGVS 3' rule: shift right
                ins, pos = ins[1:] + ref[pos], pos + 1
            k = len(ins)
            if pos >= k and ref[pos - k:pos] == ins:
                m = Mutation("duplication", pos - k + 1, pos, ins, ins)
            else:
                m = Mutation("insertion", pos, pos, "", ins)
        elif len(seg_r) == 1 and len(seg_m) == 1:
            m = Mutation("substitution", start, start, seg_r, seg_m)
        elif len(seg_r) == len(seg_m) and S.reverse_complement(seg_r) == seg_m:
            m = Mutation("inversion", start, start + len(seg_r) - 1, seg_r, seg_m)
        else:
            m = Mutation("delins", start, start + max(len(seg_r), 1) - 1, seg_r, seg_m)
        m.hgvs = hgvs_of(m)
        out.append(m)
        rpos += len(seg_r)
        i = j
    return _merge_substitutions(out)


def _gapped(aln) -> tuple[str, str]:
    lines = format(aln, "fasta").split("\n")
    seqs, cur = [], []
    for ln in lines:
        if ln.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        elif ln:
            cur.append(ln)
    if cur:
        seqs.append("".join(cur))
    return seqs[0], seqs[1]


def _find_inversion(ref: str, mut: str) -> Mutation | None:
    diffs = [i for i, (a, b) in enumerate(zip(ref, mut)) if a != b]
    if len(diffs) < 2:
        return None
    a, b = diffs[0], diffs[-1] + 1
    # widen to the largest segment that is exactly reverse-complemented
    for lo in range(a, max(-1, a - 30), -1):
        for hi in range(b, min(len(ref), b + 30) + 1):
            if S.reverse_complement(ref[lo:hi]) == mut[lo:hi] and ref[:lo] == mut[:lo] and ref[hi:] == mut[hi:]:
                m = Mutation("inversion", lo + 1, hi, ref[lo:hi], mut[lo:hi])
                m.hgvs = hgvs_of(m)
                return m
    return None


def _merge_substitutions(ms: list[Mutation]) -> list[Mutation]:
    return ms


# ---------------------------------------------------------------------------------------------
# Translocation
# ---------------------------------------------------------------------------------------------
@dataclass
class Translocation:
    der_a: str
    der_b: str
    break_a: int
    break_b: int
    fusion: str                    # der(A) read from A's start codon
    fusion_protein: str
    in_frame: bool
    partner_frame_note: str


def translocate(a: str, b: str, break_a: int, break_b: int, table_id: int = 1) -> Translocation:
    """Reciprocal translocation: the parts after the breakpoints swap places.
    der(A) = A[1..break_a] + B[break_b+1..]; der(B) = B[1..break_b] + A[break_a+1..]."""
    a, b = S.to_dna(a), S.to_dna(b)
    der_a = a[:break_a] + b[break_b:]
    der_b = b[:break_b] + a[break_a:]
    in_frame = (break_a % 3) == (break_b % 3)
    fusion_protein = _protein(der_a, table_id)
    note = ("The breakpoints sit at the same position within a codon, so the fusion keeps partner B's reading frame "
            "and the second half of the fusion protein is B's protein." if in_frame else
            "The breakpoints sit at different positions within a codon, so the partner is read out of frame and the "
            "fusion protein usually hits a stop codon.")
    return Translocation(der_a, der_b, break_a, break_b, der_a, fusion_protein, in_frame, note)


def inversion_effect(cds: str, a: int, b: int, table_id: int = 1) -> tuple[str, Effect]:
    m = Mutation("inversion", a, b, cds[a - 1:b], S.reverse_complement(cds[a - 1:b]))
    mut = apply(cds, m)
    return mut, effect(cds, mut, m, table_id)


# Worked examples on the human beta-globin coding sequence (HBB, c.1 = A of ATG)
HBB_EXAMPLES = {
    "c.20A>T": "Sickle cell haemoglobin (HbS): GAG to GTG, glutamic acid to valine. Traditionally 'Glu6Val' "
               "because older numbering skips the first methionine.",
    "c.19G>A": "Haemoglobin C (HbC): GAG to AAG, glutamic acid to lysine (traditionally 'Glu6Lys').",
    "c.79G>A": "Haemoglobin E (HbE): GAG to AAG at codon 27 (traditionally 'Glu26Lys'). It also activates a cryptic "
               "5' splice site in exon 1, which lowers HbE output; the ML splice model flags this as a donor gain.",
    "c.9T>C": "A silent change: CAT and CAC both code for histidine.",
    "c.118C>T": "Beta-zero thalassaemia, 'codon 39' nonsense mutation: CAG (glutamine) becomes the stop codon TAG.",
    "c.126_129del": "Beta-thalassaemia 'codons 41/42 (-TTCT)': four bases are lost, shifting the reading frame.",
    "c.20_21insT": "A one-base insertion (illustrative) that shifts the reading frame.",
    "c.25_27del": "An in-frame deletion (illustrative): one whole codon is removed.",
    "c.11_16dup": "A six-base duplication (illustrative): two codons' worth of bases are repeated.",
    "c.40_45inv": "An inversion (illustrative): six bases are flipped to their reverse complement.",
}


# ---------------------------------------------------------------------------------------------
# Many mutations at once
# ---------------------------------------------------------------------------------------------
def normalise(cds: str, m: Mutation) -> tuple[Mutation, str]:
    """Rewrite a typed change the way HGVS wants it (3' rule, dup instead of ins, simplest form).

    Works by applying the change and comparing the result with the reference, so the answer is the same
    as for the 'Compare two sequences' tool. Returns the mutation and a note when the notation changed."""
    m.hgvs = m.hgvs or hgvs_of(m)
    if m.kind == "inversion":
        return m, ""
    try:
        found = detect(cds, apply(cds, m))
    except BadNotation:
        return m, ""
    if len(found) != 1 or found[0].hgvs == m.hgvs:
        return m, ""
    n = found[0]
    why = ("HGVS places a deletion or insertion as far towards the 3' end as possible"
           if n.kind == m.kind and n.kind in ("deletion", "insertion", "duplication") else
           "an insertion that copies the bases just before it is written as a duplication"
           if n.kind == "duplication" else "HGVS uses the simplest description of the same change")
    return n, f"{m.hgvs} is written {n.hgvs} in HGVS: {why}. Both give the same sequence."


def classify_hgvs(cds: str, text: str, table_id: int = 1) -> dict:
    try:
        m = parse_hgvs(text, cds)
    except BadNotation as e:
        return {"input": text, "type": "Not analysed", "effect": "", "protein_change": "", "status": str(e)}
    m, note = normalise(cds, m)
    mut = apply(cds, m)
    e = effect(cds, mut, m, table_id)
    return {"input": text, "type": m.kind, "effect": e.label, "dna_change": e.dna_change,
            "protein_change": e.protein_change,
            "protein_length": "unknown" if e.label == "Start-loss" else max(0, len(e.mut_protein.rstrip("*"))),
            "status": "Analysed" + (f". {note}" if note else "")}


def classify_sequence(cds: str, rid: str, mutant: str, table_id: int = 1) -> dict:
    try:
        ms = detect(cds, mutant)
    except BadNotation as e:
        return {"input": rid, "type": "Not analysed", "effect": "", "status": str(e)}
    if not ms:
        return {"input": rid, "type": "none", "effect": "No change", "dna_change": "", "protein_change": "p.(=)",
                "status": "Identical to the reference"}
    e = effect(cds, S.to_dna(mutant), ms[0] if len(ms) == 1 else None, table_id)
    label = e.label if len(ms) == 1 else f"{e.label} (combined effect of {len(ms)} changes)"
    return {"input": rid, "type": ", ".join(sorted({m.kind for m in ms})), "effect": label,
            "dna_change": "; ".join(m.hgvs for m in ms[:6]) + (" …" if len(ms) > 6 else ""),
            "protein_change": e.protein_change,
            "protein_length": "unknown" if e.label == "Start-loss" else max(0, len(e.mut_protein.rstrip("*"))),
            "status": "Analysed"}
