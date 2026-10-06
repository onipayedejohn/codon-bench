"""App-wide settings: ownership details, version and limits. Edit AUTHOR to update the footer."""

APP_NAME = "Codon Bench"
TAGLINE = "Sequence, mutation and machine-learning analysis for DNA, RNA and proteins"
VERSION = "2.0.0"
RELEASED = "2026"

AUTHOR = {
    "name": "Onipayede John Kwaku",
    "role": "AI/ML in Bioinformatics",
    "github": "https://github.com/onipayedejohn",
    "linkedin": "",            # add your LinkedIn profile URL to show it in the footer
    "email": "",               # add a public contact email to show it in the footer
    "repository": "https://github.com/onipayedejohn/codon-bench",
}
LICENSE = "MIT License"

# Upload limits. 50 MB holds a bacterial genome, a large multi-FASTA or a VCF of several hundred
# thousand variants, while keeping memory use safe on small cloud instances.
MAX_UPLOAD_MB = 50
MAX_IMAGE_MB = 10
MAX_PDF_MB = 20

CITATION = (f"Onipayede, J. K. ({RELEASED}). {APP_NAME}: sequence, mutation and machine-learning analysis for DNA, "
            f"RNA and proteins (Version {VERSION}) "
            f"[Software]. {AUTHOR['repository']}")
