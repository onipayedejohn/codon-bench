"""Regenerate the sample files in data/samples from the reference sequences.

Run from the project folder:  python scripts/make_samples.py
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from codonbench import demos, genome as G  # noqa: E402

SAMPLES = ROOT / "data" / "samples"


def wrap(s, n=60):
    return "\n".join(s[i:i + n] for i in range(0, len(s), n))


def write_vcf(path: Path, chrom: str, length: int, rows: list[tuple], source: str):
    head = ["##fileformat=VCFv4.2", f"##source={source}", f"##contig=<ID={chrom},length={length}>",
            '##INFO=<ID=NOTE,Number=1,Type=String,Description="What the variant is known for">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
    body = [f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\tPASS\tNOTE={note.replace(' ', '_')}"
            for pos, vid, ref, alt, note in rows]
    path.write_text("\n".join(head + body) + "\n")


def main():
    seq_dir, var_dir, ref_dir, bad_dir = (SAMPLES / d for d in ("sequences", "variants", "references", "invalid"))
    # ---- single sequences in every supported format ------------------------------------------
    (seq_dir / "hbb_coding_sequence.fasta").write_text(f">HBB human beta-globin coding sequence (NM_000518.5 CDS)\n"
                                                       f"{wrap(demos.HBB_CDS)}\n")
    (seq_dir / "hbb_mrna.fa").write_text(f">HBB mRNA coding region\n{wrap(demos.HBB_CDS.replace('T', 'U'))}\n")
    (seq_dir / "hbb_plain_letters.txt").write_text(demos.HBB_CDS.lower() + "\n")
    (seq_dir / "insulin_precursor_protein.faa").write_text(f">sp|P01308|INS_HUMAN Insulin precursor\n"
                                                           f"{wrap(demos.INSULIN)}\n")
    (seq_dir / "hbb_first_120_bases.gb").write_text(demos.SINGLE["genbank"][1] + "\n")
    raw = (seq_dir / "mixed_sequences.fasta").read_bytes()
    (seq_dir / "mixed_sequences.fasta.gz").write_bytes(gzip.compress(raw, mtime=0))
    cov = G.load((ref_dir / "sars_cov_2_reference.gb").read_text())[0]
    (seq_dir / "sars_cov_2_reference_genome.fna").write_text(f">MN908947 SARS-CoV-2 Wuhan-Hu-1 complete genome\n"
                                                              f"{wrap(cov.seq)}\n")
    (ref_dir / "hbb_coding_sequence.fasta").write_text((seq_dir / "hbb_coding_sequence.fasta").read_text())

    # ---- SARS-CoV-2: lineage-defining mutations of Alpha (B.1.1.7) and the B.1 ancestor ----------
    sars = [
        (241, "", "C", "T", "5' UTR, B.1 marker"), (3037, "", "C", "T", "ORF1a F924F synonymous, B.1 marker"),
        (3267, "", "C", "T", "ORF1a T1001I, Alpha"), (5388, "", "C", "A", "ORF1a A1708D, Alpha"),
        (6954, "", "T", "C", "ORF1a I2230T, Alpha"),
        (11287, "", "GTCTGGTTTT", "G", "ORF1a SGF 3675-3677 deletion, Alpha"),
        (14408, "", "C", "T", "ORF1b P314L (RdRp P323L), B.1 marker"),
        (21764, "", "ATACATG", "A", "S H69-V70 deletion, Alpha"), (21990, "", "TTTA", "T", "S Y144 deletion, Alpha"),
        (23063, "", "A", "T", "S N501Y, Alpha"), (23271, "", "C", "A", "S A570D, Alpha"),
        (23403, "", "A", "G", "S D614G, B.1 marker"), (23604, "", "C", "A", "S P681H, Alpha"),
        (23709, "", "C", "T", "S T716I, Alpha"), (24506, "", "T", "G", "S S982A, Alpha"),
        (24914, "", "G", "C", "S D1118H, Alpha"), (27972, "", "C", "T", "ORF8 Q27 stop, Alpha"),
        (28048, "", "G", "T", "ORF8 R52I, Alpha"), (28111, "", "A", "G", "ORF8 Y73C, Alpha"),
        (28881, "", "GGG", "AAC", "N R203K and G204R, B.1.1"), (28977, "", "C", "T", "N S235F, Alpha"),
    ]
    write_vcf(var_dir / "sars_cov_2_alpha_variants.vcf", "MN908947", len(cov.seq), sars,
              "Mutations of SARS-CoV-2 Alpha (B.1.1.7) and its B.1 and B.1.1 ancestors against Wuhan-Hu-1 (not exhaustive)")
    q = cov.seq
    for pos, _, ref, alt, _ in sorted(sars, reverse=True):          # apply right to left
        assert q[pos - 1:pos - 1 + len(ref)] == ref, (pos, ref)
        q = q[:pos - 1] + alt + q[pos - 1 + len(ref):]
    q = "N" * 40 + q[40:-50]                                       # ragged ends, as in real consensus genomes
    (var_dir / "sars_cov_2_alpha_like_consensus.fasta").write_text(
        ">alpha_like_consensus Constructed from Wuhan-Hu-1 by applying the 21 variants in "
        "sars_cov_2_alpha_variants.vcf; ends masked with N\n" + wrap(q) + "\n")

    # ---- human mitochondrial DNA, rCRS (NC_012920.1) ----------------------------------------------
    mt = G.load((ref_dir / "human_mitochondrion_rCRS.gb").read_text())[0]
    mito = [
        (73, "", "A", "G", "D-loop HVS2, common polymorphism"), (263, "", "A", "G", "D-loop, common polymorphism"),
        (310, "", "T", "TC", "D-loop poly-C insertion (315.1C), common"),
        (1555, "", "A", "G", "MT-RNR1, aminoglycoside-induced hearing loss"),
        (2706, "", "A", "G", "MT-RNR2, 2706G marks non-H haplogroups (rCRS carries the H allele A)"), (3243, "", "A", "G", "MT-TL1, MELAS"),
        (3460, "", "G", "A", "MT-ND1 A52T, LHON"), (4216, "", "T", "C", "MT-ND1 Y304H, haplogroup JT"),
        (7028, "", "C", "T", "MT-CO1 A375A synonymous, 7028T marks non-H haplogroups (rCRS carries the H allele C)"), (8344, "", "A", "G", "MT-TK, MERRF"),
        (8993, "", "T", "G", "MT-ATP6 L156R, NARP"), (11719, "", "G", "A", "MT-ND4 G320G synonymous, haplogroup marker (rCRS carries G)"),
        (11778, "", "G", "A", "MT-ND4 R340H, LHON"), (14484, "", "T", "C", "MT-ND6 M64V, LHON"),
        (15326, "", "A", "G", "MT-CYB T194A, common polymorphism"),
    ]
    for pos, _, ref, alt, _ in mito:
        assert mt.seq[pos - 1:pos - 1 + len(ref)] == ref, (pos, ref, mt.seq[pos - 1:pos + 2])
    write_vcf(var_dir / "human_mtdna_variants.vcf", "NC_012920.1", len(mt.seq), mito,
              "Well-characterised human mitochondrial variants against the rCRS")
    with open(var_dir / "human_mtdna_variants_positions.csv", "w") as fh:
        fh.write("id,pos,ref,alt,note\n")
        for pos, _, ref, alt, note in mito:
            fh.write(f"m.{pos},{pos},{ref},{alt},\"{note}\"\n")

    # ---- ADAMTS1 genomic region: splice-site and coding variants -----------------------------------
    ad = G.load((ref_dir / "human_ADAMTS1_genomic.gb").read_text())[0]
    cds0 = ad.cds[0]
    first_exon_end = sorted(cds0.parts)[0][1]                 # 0-based end of the first coding exon part
    start = sorted(cds0.parts)[0][0]
    found = {}
    for pos in range(start + 31, first_exon_end - 40):         # pick real missense and synonymous changes
        for alt in "ACGT":
            if alt == ad.seq[pos - 1]:
                continue
            e = G.annotate(ad, G.Variant(ad.id, pos, ad.seq[pos - 1], alt))["effect"]
            if e in ("Missense", "Silent", "Nonsense") and e not in found:
                found[e] = (pos, ad.seq[pos - 1], alt)
        if len(found) == 3:
            break
    rows = [("exon1_missense", *found["Missense"]), ("exon1_synonymous", *found["Silent"]),
            ("exon1_nonsense", *found["Nonsense"]),
            ("intron1_donor_plus1", first_exon_end + 1, ad.seq[first_exon_end], "A" if ad.seq[first_exon_end] != "A" else "C"),
            ("intron1_donor_plus5", first_exon_end + 5, ad.seq[first_exon_end + 4], "G" if ad.seq[first_exon_end + 4] != "G" else "A"),
            ("intron1_acceptor_minus2", 2923, "A", "G"), ("intron1_deep", 2000, ad.seq[1999], "A" if ad.seq[1999] != "A" else "G")]
    with open(var_dir / "adamts1_genomic_variants.csv", "w") as fh:
        fh.write("id,pos,ref,alt\n")
        for vid, pos, ref, alt in rows:
            fh.write(f"{vid},{pos},{ref},{alt}\n")
        fh.write(f"exon2_frameshift_del,3000,{ad.seq[2999:3001]},{ad.seq[2999]}\n")
        fh.write(f"exon3_inframe_del,3650,{ad.seq[3649:3653]},{ad.seq[3649]}\n")

    # ---- HBB list as Excel too -------------------------------------------------------------------
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "variants"
    for line in (var_dir / "hbb_variants_hgvs.csv").read_text().splitlines():
        ws.append(line.split(","))
    wb.save(var_dir / "hbb_variants_hgvs.xlsx")

    # ---- inputs that must be refused --------------------------------------------------------------
    (bad_dir / "prose_not_a_sequence.txt").write_text(
        "Quarterly sales grew by twelve percent compared with last year, driven by strong demand in the "
        "northern region. The board will meet on Friday to review the budget.\n")
    png = (SAMPLES / "invalid" / "not_related_poster.png").read_bytes()
    (bad_dir / "image_renamed_as.fasta").write_bytes(png)
    print("samples written")


if __name__ == "__main__":
    main()
