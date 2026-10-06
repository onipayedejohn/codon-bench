"""Example inputs. The HBB coding sequence translates exactly to UniProt P68871 (human beta-globin);
the insulin precursor is UniProt P01308. Other examples are short synthetic sequences."""

HBB_CDS = ("ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTG"
           "GTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAG"
           "TGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCC"
           "TGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGG"
           "TGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCACTAA")
INSULIN = ("MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSIC"
           "SLYQLENYCN")
HBB_PROTEIN = ("MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHV"
               "DPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH")


def _wrap(s, n=60):
    return "\n".join(s[i:i + n] for i in range(0, len(s), n))


SINGLE = {
    "hbb": ("Human beta-globin gene, coding sequence (DNA)", f">HBB beta-globin coding sequence\n{_wrap(HBB_CDS)}"),
    "hbb_mrna": ("Beta-globin mRNA (RNA)", f">HBB mRNA\n{_wrap(HBB_CDS.replace('T', 'U'))}"),
    "synthetic": ("Short synthetic ORF (DNA)", ">Synthetic open reading frame\nATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"),
    "template": ("Template strand to transcribe (DNA)", ">Template strand, written 5'->3' (RNA polymerase reads it 3'->5')\n"
                 "CTATCGGGCACCCTTTCAGCGGCCCATTACAATGGCCAT"),
    "insulin": ("Human insulin precursor (protein)", f">sp|P01308|INS_HUMAN Insulin\n{_wrap(INSULIN)}"),
    "genbank": ("GenBank-style record (DNA)", "LOCUS       HBB_START   120 bp    DNA\nDEFINITION  Beta-globin, first "
                "120 bases of the coding sequence.\nORIGIN\n" +
                "\n".join(f"{i + 1:>9} " + " ".join(HBB_CDS[i:i + 60][j:j + 10] for j in range(0, 60, 10)).lower()
                          for i in range(0, 120, 60)) + "\n//"),
}

_GENOMIC = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "samples" / "sequences" / \
    "human_ADAMTS1_genomic_region.fasta"
if _GENOMIC.exists():
    SINGLE["adamts1"] = ("Human ADAMTS1 gene with introns, 10 kb (DNA, for splice prediction)", _GENOMIC.read_text())

