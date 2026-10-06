"""Variant annotation on real genomes, VCF parsing, genome comparison and the ML models."""
import csv
from pathlib import Path

import numpy as np
import pytest

from codonbench import files as F
from codonbench import genome as G
from codonbench import ml

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def ref(name):
    return G.load((SAMPLES / "references" / name).read_text())[0]


COV = ref("sars_cov_2_reference.gb")
MT = ref("human_mitochondrion_rCRS.gb")
AD = ref("human_ADAMTS1_genomic.gb")


@pytest.mark.parametrize("change,short,effect", [
    ("A23403G", "S:D614G", "Missense"), ("A23063T", "S:N501Y", "Missense"), ("G23012A", "S:E484K", "Missense"),
    ("C23604A", "S:P681H", "Missense"), ("C14408T", "ORF1b:P314L", "Missense"),
    ("C3037T", "ORF1a:F924F", "Silent"), ("C27972T", "ORF8:Q27*", "Nonsense"),
])
def test_sars_cov_2_known_mutations(change, short, effect):
    a = G.annotate(COV, G.parse_genomic_change(change, COV))
    assert (a["protein_short"], a["effect"]) == (short, effect)


def test_spike_deletion_is_3prime_shifted():
    a = G.annotate(COV, G.Variant(COV.id, 21764, "ATACATG", "A"))
    assert a["hgvs_g"] == "g.21766_21771del" and a["hgvs_p"] == "p.His69_Val70del"


@pytest.mark.parametrize("change,gene,protein", [
    ("m.11778G>A", "ND4", "p.Arg340His"), ("m.8993T>G", "ATP6", "p.Leu156Arg"),
    ("m.14484T>C", "ND6", "p.Met64Val"),            # ND6 is on the minus strand
    ("m.3460G>A", "ND1", "p.Ala52Thr"),
])
def test_mitochondrial_variants_use_code_2_and_strand(change, gene, protein):
    a = G.annotate(MT, G.parse_genomic_change(change, MT))
    assert (a["gene"], a["hgvs_p"]) == (gene, protein)
    assert MT.table == 2 and MT.prefix == "m"


def test_mitochondrial_trna_and_insertion():
    assert "tRNA" in G.annotate(MT, G.parse_genomic_change("m.3243A>G", MT))["effect"]
    assert G.annotate(MT, G.Variant(MT.id, 310, "T", "TC"))["hgvs_g"] == "m.315dup"


def test_intronic_and_splice_site_notation():
    a = G.annotate(AD, G.Variant(AD.id, 1386, "G", "A"))
    assert a["effect"] == "Splice donor site" and a["hgvs_c"] == "c.730+1G>A"
    b = G.annotate(AD, G.Variant(AD.id, 2923, "A", "G"))
    assert b["effect"] == "Splice acceptor site" and b["hgvs_c"] == "c.731-2A>G"


def test_reference_mismatch_is_reported():
    a = G.annotate(COV, G.Variant(COV.id, 23403, "C", "G"))
    assert "does not match" in a["status"]


def test_vcf_parsing_and_consensus_comparison():
    vs, problems = G.parse_vcf((SAMPLES / "variants" / "sars_cov_2_alpha_variants.vcf").read_text())
    assert len(vs) == 21 and not problems
    q = (SAMPLES / "variants" / "sars_cov_2_alpha_like_consensus.fasta").read_text().split("\n", 1)[1]
    found = G.compare_genomes(COV, q)
    assert sorted(G.annotate(COV, v)["hgvs_g"] for v in found) == sorted(G.annotate(COV, v)["hgvs_g"] for v in vs)


def test_spectrum_and_ti_tv():
    vs, _ = G.parse_vcf((SAMPLES / "variants" / "human_mtdna_variants.vcf").read_text())
    s = G.summarise([G.annotate(MT, v) for v in vs])
    assert s["snv"] == 14 and s["tv"] == 1 and sum(s["spectrum"].values()) == 14


def test_vcf_file_family():
    p = SAMPLES / "variants" / "human_mtdna_variants.vcf"
    assert F.sniff(str(p), p.name, p.stat().st_size).family == "variants"


def test_upload_limit_is_50_mb():
    assert F.MAX_BYTES == 50 * 1024 * 1024


# ---- machine learning ------------------------------------------------------------------------
pytestmark_ml = pytest.mark.skipif(not ml.available(), reason="models not trained")


@pytestmark_ml
def test_model_card_meets_reported_performance():
    card = ml.model_card()
    for name in ("splice_donor", "splice_acceptor", "coding_potential"):
        assert card["models"][name]["test_human"]["auroc"] > 0.95
        assert card["models"][name]["onnx_max_abs_difference"] < 1e-4


@pytestmark_ml
def test_real_adamts1_splice_sites_score_higher_than_decoys():
    exons = sorted(AD.features[1].parts)                  # mRNA feature
    donors = [b for _, b in exons[:-1]]                   # 0-based first intron base
    pos, sc = ml.score_sites(AD.seq, "donor")
    d = dict(zip(pos, sc))
    true = np.array([d[p] for p in donors if p in d])
    decoy = np.array([s for p, s in d.items() if p not in donors])
    assert true.mean() > 0.5 and np.median(decoy) < 0.05


@pytestmark_ml
def test_donor_plus1_variant_destroys_the_site():
    a = G.annotate(AD, G.Variant(AD.id, 1386, "G", "A"), splice=True)
    assert a["splice_detail"]["donor"]["loss"] <= -0.5


@pytestmark_ml
def test_coding_potential_separates_cds_from_intron():
    cds = AD.feature_seq(AD.cds[0])
    intron = AD.seq[1500:2800]
    assert ml.coding_probability(cds) > 0.8 and ml.coding_probability(intron) < 0.3
