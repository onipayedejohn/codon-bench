# Training the models

The app ships with three trained models in `models/` (ONNX format) and their evaluation in
`models/model_card.json` and `models/curves.json`. This folder rebuilds them from public data.

```bash
pip install -r requirements.txt -r training/requirements-train.txt
python training/download_data.py      # about 70 MB, checksums verified
python training/build_datasets.py     # about 30 seconds
python training/train_models.py       # about 20 minutes on 2 CPU cores
```

## Design

| | Splice donor | Splice acceptor | Coding potential |
|---|---|---|---|
| Question | Is this GT the start of an intron? | Is this AG the end of an intron? | Is this 150-base window protein-coding? |
| Positives | every canonical intron start of every transcript | every canonical intron end | windows of spliced CDS, all three frames |
| Negatives | other GT in the same genes, 25 per positive | other AG in the same genes, 25 per positive | introns, intergenic DNA (either strand), UTRs, lncRNA exons, equal in number |
| Features | one-hot bases in an 80-base window, 3-mers on each side, GC, and the composition of 160 flanking bases on each side | one-hot bases in a 100-base window, 3-mers, polypyrimidine tract, branch-point region, AG exclusion zone, flanking composition | stop-codon rate in each frame, codon usage and position-specific composition of the best frame, 1- to 3-mers, Fickett TESTCODE, period-3 Fourier power |

- **Training data:** mouse chromosome 19, GENCODE M31.
- **Test data:** human chromosome 21, GENCODE. It is never used for fitting, early stopping or threshold choice.
- **Model selection:** 5-fold cross-validation grouped by gene, logistic regression baseline against LightGBM.
- **Threshold:** maximises F1 on mouse out-of-fold predictions.
- **Robustness:** 10% of training windows lose part of one flank (filled with N), so sites near the ends of a short sequence are still scored sensibly.
- **Export:** LightGBM to ONNX with onnxmltools; the script checks that both give the same probabilities.
- **Extra splice test:** a SpliceAI-style top-k accuracy, scoring every GT or AG in each human gene.
- **Extra coding test:** whole human transcripts, coding sequences against lncRNAs, scored with the same code path the app uses.

`codonbench/ml.py` holds the feature code for both training and the app, so they cannot drift apart.
