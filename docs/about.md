Codon Bench is a web application for analysing DNA, RNA and protein sequences and the variants in them. It combines classical sequence analysis with machine-learning models trained on real genome annotation, and it works on sequences from any organism.

### What you can do

**Analyse a sequence.** Paste or upload DNA, RNA or protein. The app identifies the molecule, shows both strands with their base pairing, and walks codon by codon through replication, transcription, translation and reverse transcription, including each tRNA anticodon. It reports composition, GC content, melting temperature, codon usage, open reading frames in all six frames and protein properties (molecular weight, isoelectric point, hydropathy, instability and more).

**Predict with machine learning.** Three models score any DNA sequence: the probability that a stretch is protein-coding, and the probability that each GT or AG is a real splice donor or acceptor. Every open reading frame also gets a coding probability. How well the models perform, and how that was measured, is on the Model performance page.

**Annotate mutations in any genome.** Choose a built-in reference (human beta-globin, a human gene with introns, the human mitochondrial genome or SARS-CoV-2) or load your own as FASTA or GenBank. Single variants, whole VCF files, CSV or Excel tables, mutant sequences and consensus genomes are all placed in their gene, named in HGVS notation (c., g., m. and p.) and classified: silent, missense, nonsense, stop-loss, start-loss, frameshift, in-frame insertion or deletion, duplication, inversion, splice-site, intronic, UTR or intergenic. Each gene uses its own genetic code, so mitochondrial and other non-standard genes translate correctly. Batches report the transition/transversion ratio, the single-base substitution spectrum and variants per gene. When the reference is a eukaryotic nuclear gene, every variant also gets an ML splice-impact score (gain or loss of donor and acceptor strength nearby).

**Model structural changes.** Inversions, reciprocal translocations (with the fusion protein they make), duplications and deletions are drawn to scale and checked base by base.

**Read sequences from documents.** Photos, scans and PDFs are read with optical character recognition. Each line is classified as a header, sequence, table row or other text, sequence lines are joined, and you can correct the result before analysing it.

**Keep a record.** The History page lists everything analysed in the session and downloads it as CSV.

### Inputs

FASTA (.fasta, .fa, .fna, .faa), GenBank (.gb, .gbk, .gbff), plain text, VCF, CSV, TSV and Excel files, gzip-compressed or not, up to 50 MB; images up to 10 MB and PDFs up to 20 MB. Files are checked by their contents as well as their name. Anything that is not a biological sequence or variant, such as ordinary text, an unrelated table or an image renamed as a sequence file, is refused with the reason. The check is a heuristic: a random string of amino-acid letters can still pass as a protein.

### Limits

- Codon Bench is a research tool. It is not a clinical or diagnostic tool, and its predictions are not evidence of pathogenicity.
- Mutation effects are predicted from sequence. Protein structure, gene regulation, population frequency and clinical significance are not assessed.
- The splice models read about 400 bases around each site, so they miss long-range signals. Treat their scores as a guide to prioritise variants for laboratory testing.
- Protein properties are calculated estimates, not measurements. OCR can misread a letter, so check sequences read from images against the original.
- Uploads are processed on the server and are not saved to a database. Do not upload patient-identifiable data.

### Built with

Shiny for Python, Biopython, LightGBM (trained offline) and ONNX Runtime (inference), Plotly, pandas, openpyxl, RapidOCR and pypdfium2. Reference data: GENCODE annotation of mouse chromosome 19 and human chromosome 21 (via nf-core/test-datasets), the human mitochondrial rCRS (NC_012920.1), SARS-CoV-2 Wuhan-Hu-1 (MN908947), human beta-globin (NM_000518.5; its coding sequence translates to UniProt P68871) and the human insulin precursor (UniProt P01308).
