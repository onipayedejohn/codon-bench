# Data

Every file here is either public reference data or derived from it. Nothing comes from patients.

```
data/
├── samples/                 example inputs for every demo button and every supported file type
│   ├── sequences/           single and multiple sequences: FASTA, FA, FNA, FAA, TXT, GenBank, CSV, XLSX, .gz
│   ├── references/          reference genomes and genes for the mutation analyser (FASTA and GenBank)
│   ├── variants/            variant files: VCF, CSV, XLSX, mutant FASTA and a consensus genome
│   ├── documents/           images and a PDF for the document reader (OCR)
│   └── invalid/             inputs the app must refuse (prose, an unrelated table, a disguised image)
└── training_samples/        300-row samples of each machine-learning dataset, with the dataset sizes
```

## References

| File | What it is | Source |
|---|---|---|
| `references/hbb_coding_sequence.fasta` | Human beta-globin coding sequence, 444 bp | NM_000518.5 CDS; translates to UniProt P68871 |
| `references/human_ADAMTS1_genomic.gb` | Human ADAMTS1 gene, 10,055 bp with 9 exons and 8 introns, on the transcript strand | GRCh38 chr21, GENCODE transcript ENST00000284984.8 |
| `references/human_mitochondrion_rCRS.gb` | Human mitochondrial genome, 16,569 bp, 13 protein-coding and 24 RNA genes, genetic code 2 | NC_012920.1 (rCRS) |
| `references/sars_cov_2_reference.gb` | SARS-CoV-2 Wuhan-Hu-1, 29,903 bp, 12 coding sequences | MN908947, as distributed by Nextstrain |

## Variants

| File | Contents |
|---|---|
| `variants/sars_cov_2_alpha_variants.vcf` | 21 mutations carried by SARS-CoV-2 Alpha (B.1.1.7): its defining mutations plus markers inherited from B.1 and B.1.1. The list is not exhaustive (N:D3L and several synonymous markers are left out). It includes the spike H69-V70 and Y144 deletions, N501Y, D614G and P681H. HGVS places the Y144 deletion at Tyr145 (3' rule) because residues 144 and 145 are both tyrosine. |
| `variants/sars_cov_2_alpha_like_consensus.fasta` | A genome built by applying those 21 mutations to Wuhan-Hu-1, with N-masked ends. Comparing it with the reference recovers all 21, with the two spike deletions written in their 3'-shifted HGVS positions. |
| `variants/human_mtdna_variants.vcf` and `_positions.csv` | 15 well-characterised mitochondrial variants: m.3243A>G (MELAS), m.11778G>A, m.3460G>A and m.14484T>C (LHON), m.8993T>G (NARP), m.8344A>G (MERRF), m.1555A>G (aminoglycoside hearing loss) and haplogroup markers. The rCRS belongs to haplogroup H, so alleles such as 7028T and 2706G mark non-H lineages. |
| `variants/hbb_variants_hgvs.csv` and `.xlsx` | Beta-globin variants in HGVS c. notation, including HbS (c.20A>T), HbC (c.19G>A), codon 39 nonsense and codons 41/42 frameshift, plus deliberately invalid rows. |
| `variants/hbb_mutant_sequences.fasta` | Mutant beta-globin sequences, one identical copy and one protein record that must be refused. |
| `variants/adamts1_genomic_variants.csv` | Coding, splice-donor, splice-acceptor, splice-region and deep intronic variants in ADAMTS1, as pos, ref and alt columns. |

## Machine-learning datasets

`training/build_datasets.py` creates them from mouse chromosome 19 (GRCm39, GENCODE M31) and human chromosome 21 (GRCh38, GENCODE), downloaded by `training/download_data.py`. The processed human test tables and the coding tables are in `training/processed/`. The two mouse splice tables (25 MB together) are left out to keep the repository small; `python training/build_datasets.py` rebuilds them in about 30 seconds. `training_samples/` holds small samples of every table for a quick look.

| Column | Meaning |
|---|---|
| `species` | mouse (training) or human (test) |
| `gene_id` | Ensembl gene ID; cross-validation groups by it |
| `chrom`, `strand`, `pos0` | location of the site (0-based, splice datasets) |
| `source` | coding datasets: CDS, intron, intergenic, UTR or lncRNA |
| `label` | 1 for a real splice site or coding window, 0 otherwise |
| `window` | the DNA sequence the features are computed from, on the transcript strand |
