"""Download the genomes and annotations used to train and test the models.

Sources are the public nf-core test-datasets repository (branch `modules`):
  Mouse chromosome 19, GRCm39, with GENCODE M31 (Ensembl 108) annotation   -> training and validation
  Human chromosome 21, GRCh38, with a GENCODE annotation subset             -> held-out test

Run from the project folder:  python training/download_data.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/nf-core/test-datasets/modules/data/genomics"
FILES = {
    "mm_chr19.fa.gz": (f"{BASE}/mus_musculus/genome/chr19.fa.gz",
                       "aa64f5018981c0cb1463ce1bbc33ed86317d25aa611abde34fe52ed068975ada"),
    "mm_chr19.gtf.gz": (f"{BASE}/mus_musculus/genome/chr19.filtered.gtf.gz",
                        "4374b4656d0ab6b3e1cd0e73057d00dec0c9715b4f00bba32f15747eb09175ce"),
    "hs_chr21.fa": (f"{BASE}/homo_sapiens/genome/chr21/sequence/genome.fasta",
                    "634829fe7eba3074f481818a448f7e504e19bdcb992d3388aad4de034739e0b2"),
    "hs_chr21.gtf": (f"{BASE}/homo_sapiens/genome/chr21/sequence/chr21_gencode.gtf",
                     "41372960a6f4d57bd41249f85051e16615ccb5f48dfe50a41aa9082d6f07c0a0"),
}
RAW = Path(__file__).resolve().parent / "raw"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    RAW.mkdir(exist_ok=True)
    for name, (url, digest) in FILES.items():
        dest = RAW / name
        if not dest.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, dest)
        ok = sha256(dest) == digest
        print(f"{name}: {'checksum OK' if ok else 'CHECKSUM MISMATCH'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
