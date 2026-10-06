"""Core checks: run with `pytest -q` from the project folder."""
from pathlib import Path

import pytest

from codonbench import batch as B
from codonbench import demos, docread
from codonbench import files as F
from codonbench import mutations as M
from codonbench import seq as S

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
HBB = demos.HBB_CDS


# ---- sequences and the central dogma --------------------------------------------------------
def test_hbb_translates_to_uniprot_p68871():
    assert S.translate(HBB).rstrip("*") == demos.HBB_PROTEIN


def test_transcription_and_reverse_transcription_round_trip():
    mrna = S.transcribe(HBB, "coding")
    assert mrna == HBB.replace("T", "U")
    rt = S.reverse_transcribe(mrna)
    assert rt["cdna_second_strand"] == HBB
    assert rt["cdna_first_strand"] == S.reverse_complement(HBB)


def test_template_strand_gives_same_mrna():
    template = S.reverse_complement(HBB)
    assert S.transcribe(template, "template") == S.transcribe(HBB, "coding")


def test_anticodon_is_antiparallel():
    a = S.anticodon("AUG")
    assert a["three_to_five"] == "UAC"
    assert a["five_to_three"] == "CAU"


def test_longest_orf_is_the_coding_frame():
    orfs = S.find_orfs(HBB, min_aa=30)
    assert orfs[0].frame == "+1" and orfs[0].length_aa == 147 and orfs[0].complete


def test_gc_content_hbb():
    st = S.nucleotide_stats(HBB, "DNA")
    assert st["length"] == 444
    assert round(st["gc_percent"], 2) == 56.08


# ---- input guard ------------------------------------------------------------------------------
@pytest.mark.parametrize("text,kind", [
    (HBB, "DNA"),
    (HBB.replace("T", "U"), "RNA"),
    (demos.INSULIN, "Protein"),
])
def test_detects_sequence_types(text, kind):
    assert S.parse(text).kind == kind


@pytest.mark.parametrize("text", [
    "Sales revenue Q3 2024 increased by twelve percent compared with last year",
    "Dear student, please submit your assignment by Friday.",
    "ACGU ACGT mixed",
])
def test_refuses_unrelated_text(text):
    assert S.parse(text).kind not in ("DNA", "RNA", "Protein")


def test_genbank_reads_origin_only():
    p = S.parse(demos.SINGLE["genbank"][1])
    assert p.kind == "DNA" and len(p.seq) == 120


# ---- batch files ------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["mixed_sequences.fasta", "mixed_sequences.csv", "mixed_sequences.xlsx",
                                  "mixed_sequences.fasta.gz"])
def test_batch_demo_files(name):
    path = SAMPLES / "sequences" / name
    r = B.run(str(path), name, path.stat().st_size, 1, 30, None)
    assert r.n_records == 27 and r.n_blocked >= 1


def test_batch_refuses_unrelated_table():
    path = SAMPLES / "invalid" / "not_sequences.csv"
    with pytest.raises(F.Blocked):
        B.run(str(path), path.name, path.stat().st_size, 1, 30, None)


def test_size_limit_is_enforced(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">a\nACGT\n")
    with pytest.raises(F.Blocked):
        F.sniff(str(p), p.name, F.MAX_BYTES + 1)


def test_disguised_file_is_refused(tmp_path):
    p = tmp_path / "photo.fasta"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    with pytest.raises(F.Blocked):
        F.sniff(str(p), p.name, p.stat().st_size)


# ---- document reader --------------------------------------------------------------------------
def test_text_pdf_finds_hbb_and_insulin():
    res = docread.read_pdf(str(SAMPLES / "documents" / "sequence_handout.pdf"))
    seqs = [f.sequence for f in res.found]
    assert any(s.startswith(HBB[:180]) for s in seqs)
    assert any(demos.INSULIN in s for s in seqs)


# ---- mutations --------------------------------------------------------------------------------
@pytest.mark.parametrize("hgvs,label,protein", [
    ("c.20A>T", "Missense", "p.Glu7Val"),            # HbS, traditionally Glu6Val
    ("c.9T>C", "Silent", "p.His3="),
    ("c.118C>T", "Nonsense", "p.Gln40Ter"),           # traditional codon 39
    ("c.126_129del", "Frameshift", "p.Phe42LeufsTer19"),  # traditional codons 41/42 -TTCT
    ("c.20_21insT", "Frameshift", "p.Glu7AspfsTer17"),
    ("c.25_27del", "In-frame deletion", "p.Lys9del"),
    ("c.11_16dup", "Duplication", "p.Leu4_Thr5dup"),
    ("c.1A>G", "Start-loss", "p.Met1?"),
])
def test_hbb_mutation_effects(hgvs, label, protein):
    r = M.classify_hgvs(HBB, hgvs)
    assert (r["effect"], r["protein_change"]) == (label, protein)


def test_wrong_reference_base_is_rejected():
    with pytest.raises(M.BadNotation):
        M.parse_hgvs("c.20G>T", HBB)


def test_detect_recovers_3prime_shifted_deletion():
    mut = M.apply(HBB, M.parse_hgvs("c.126_129del", HBB))
    ms = M.detect(HBB, mut)
    assert [m.hgvs for m in ms] == ["c.126_129del"]


def test_detect_finds_duplication_and_inversion():
    dup = M.apply(HBB, M.parse_hgvs("c.11_16dup", HBB))
    inv = M.apply(HBB, M.parse_hgvs("c.40_45inv", HBB))
    assert M.detect(HBB, dup)[0].kind == "duplication"
    assert M.detect(HBB, inv)[0].kind == "inversion"


def test_identical_sequence_has_no_change():
    assert M.classify_sequence(HBB, "same", HBB)["effect"] == "No change"


def test_translocation_lengths():
    b = "ATGGATACCGAGCGTAAACTGCAGTCTGGCAAACGTCCGGTTGAAGATCTGCGCTTCAAA"
    t = M.translocate(HBB, b, 30, 9)
    assert len(t.der_a) == 30 + len(b) - 9
    assert len(t.der_b) == 9 + len(HBB) - 30


@pytest.mark.parametrize("hgvs,protein", [
    ("c.22_23insTAA", "p.Glu8delinsValLys"),          # insertion that splits a codon
    ("c.17_19del", "p.Pro6_Glu7delinsGln"),            # deletion that splits a codon
    ("c.19_21delinsGAA", "p.Glu7="),                   # silent delins
    ("c.40_45inv", "p.Ala14_Leu15delinsGlnGly"),       # neighbouring changes as one delins
])
def test_protein_notation_for_codon_splitting_changes(hgvs, protein):
    assert M.classify_hgvs(HBB, hgvs)["protein_change"] == protein


@pytest.mark.parametrize("typed,hgvs", [
    ("c.124_127del", "c.126_129del"),   # 3' rule
    ("c.27_28insG", "c.27dup"),         # insertion of a copy is a duplication
    ("c.17_18delinsTT", "c.17C>T"),     # simplest form
])
def test_typed_notation_is_normalised(typed, hgvs):
    m, note = M.normalise(HBB, M.parse_hgvs(typed, HBB))
    assert m.hgvs == hgvs and note
