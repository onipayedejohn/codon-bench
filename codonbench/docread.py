"""Read sequences from images and PDFs: OCR, then document understanding.

OCR (RapidOCR, ONNX models that ship inside the pip package) turns pixels into
words with boxes and confidences. Document understanding then works out what
each line is: a FASTA header, a sequence line (with or without position
numbers), a table row that holds a sequence, or ordinary text. Consecutive
sequence lines are joined into one sequence, named after the header above them.
An image with no sequence lines is refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from . import seq as S
from .files import Blocked

MAX_SIDE = 2200
MAX_PAGES = 15
ROLE_COLORS = {"header": (74, 58, 167), "sequence": (27, 175, 122), "table": (42, 120, 214), "text": (150, 156, 168)}

_engine = None


def engine():
    global _engine
    if _engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _engine = RapidOCR()
        except (ImportError, OSError) as e:   # e.g. OpenCV missing libGL on a slim server
            from .files import Blocked
            raise Blocked("Text recognition (OCR) is not available on this server, so images and scanned PDFs "
                          f"cannot be read. PDFs that contain real text still work. Details: {e}") from e
    return _engine


@dataclass
class Line:
    text: str
    box: tuple[int, int, int, int]
    conf: float
    role: str = "text"
    page: int = 1


@dataclass
class Found:
    name: str
    sequence: str
    kind: str
    lines: int
    confidence: float          # mean OCR confidence of its lines (1.0 for PDF text)
    page: int


@dataclass
class Reading:
    doc_type: str
    found: list[Found]
    lines: list[Line]
    overlays: list[Image.Image] = field(default_factory=list)
    source: str = "OCR"          # "OCR" or "PDF text"
    pages: int = 1


# ---------------------------------------------------------------------------------------------
def load_image(path: str) -> Image.Image:
    try:
        im = Image.open(path)
        im.load()
    except Exception as e:  # noqa: BLE001
        raise Blocked("The image could not be opened. It may be damaged.") from e
    im = ImageOps.exif_transpose(im).convert("RGB")
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.Resampling.LANCZOS)
    return im


def ocr_lines(im: Image.Image, page: int = 1) -> list[Line]:
    # the angle classifier sometimes flips upright lines of lower-case sequence and reads
    # them as digits, so it is off: documents are expected upright (EXIF rotation is applied)
    res, _ = engine()(np.asarray(im), use_cls=False)
    words = []
    for box, text, conf in res or []:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        words.append(Line(text.strip(), (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), float(conf),
                          page=page))
    return group_lines(words)


def group_lines(words: list[Line]) -> list[Line]:
    """Merge OCR fragments that sit on the same printed line (left to right)."""
    words = sorted(words, key=lambda w: ((w.box[1] + w.box[3]) / 2, w.box[0]))
    lines: list[list[Line]] = []
    for w in words:
        cy = (w.box[1] + w.box[3]) / 2
        h = w.box[3] - w.box[1]
        if lines:
            last = lines[-1]
            lcy = sum((x.box[1] + x.box[3]) / 2 for x in last) / len(last)
            lh = max(x.box[3] - x.box[1] for x in last)
            if abs(cy - lcy) < 0.5 * max(h, lh):
                last.append(w)
                continue
        lines.append([w])
    out = []
    for grp in lines:
        grp.sort(key=lambda w: w.box[0])
        # a wide horizontal gap between fragments suggests table cells
        gaps = [b.box[0] - a.box[2] for a, b in zip(grp, grp[1:])]
        h = max(x.box[3] - x.box[1] for x in grp)
        sep = " | " if any(g > 2.5 * h for g in gaps) else " "
        text = sep.join(w.text for w in grp)
        box = (min(w.box[0] for w in grp), min(w.box[1] for w in grp), max(w.box[2] for w in grp),
               max(w.box[3] for w in grp))
        conf = sum(w.conf * len(w.text) for w in grp) / max(1, sum(len(w.text) for w in grp))
        out.append(Line(text, box, conf, page=grp[0].page))
    return out


# ---------------------------------------------------------------------------------------------
_POS = re.compile(r"^\s*\d+\s*")


def seq_part(text: str) -> tuple[str, str]:
    """Return (sequence letters, kind) if a line is mostly sequence, else ("", "")."""
    t = _POS.sub("", text)
    t = re.sub(r"\s+\d+\s*$", "", t)                  # trailing position numbers
    letters = re.sub(r"[\s\-\.]", "", t)
    if len(letters) < 8 or not letters.isalpha():
        return "", ""
    up = letters.upper()
    kind, _ = S.detect(up, t)
    if kind == "Unknown":
        return "", ""
    if kind == "Protein" and not (letters.isupper() or letters.islower()):
        return "", ""
    return up, kind


def classify(lines: list[Line]) -> None:
    for ln in lines:
        t = ln.text.strip()
        if t.startswith(">") or re.match(r"^(LOCUS|DEFINITION|ACCESSION)\b", t):
            ln.role = "header"
        elif " | " in t and any(seq_part(c)[0] for c in t.split(" | ")):
            ln.role = "table"
        elif seq_part(t)[0]:
            ln.role = "sequence"
        else:
            ln.role = "text"


def assemble(lines: list[Line]) -> list[Found]:
    found: list[Found] = []
    name, buf, confs, kinds, page = "", [], [], [], 1
    unnamed = 0

    def flush():
        nonlocal buf, confs, kinds, name, unnamed
        if buf:
            seq = "".join(buf)
            kind, _ = S.detect(seq)
            if kind != "Unknown":
                if not name:
                    unnamed += 1
                found.append(Found(name or f"Sequence {unnamed}", seq, kind, len(buf),
                                   float(np.mean(confs)), page))
        buf, confs, kinds = [], [], []

    for ln in lines:
        if ln.role == "header":
            flush()
            name = ln.text.lstrip(">").strip()[:80]
            page = ln.page
        elif ln.role == "sequence":
            if not buf:
                page = ln.page
            s, k = seq_part(ln.text)
            buf.append(s)
            confs.append(ln.conf)
            kinds.append(k)
        elif ln.role == "table":
            flush()
            cells = [c.strip() for c in ln.text.split(" | ")]
            label = next((c for c in cells if not seq_part(c)[0]), "")
            for c in cells:
                s, k = seq_part(c)
                if s:
                    found.append(Found(label[:80] or f"Table row {len(found) + 1}", s, k, 1, ln.conf, ln.page))
        else:
            flush()
            name = ""
    flush()
    return found


def doc_type(lines: list[Line], found: list[Found]) -> str:
    roles = [ln.role for ln in lines]
    if not found:
        return "No sequence found"
    if roles.count("table") >= max(1, roles.count("sequence")):
        return "Table of sequences"
    if any(ln.role == "header" and ln.text.startswith(">") for ln in lines):
        return "FASTA printout"
    if any(ln.role == "sequence" and _POS.match(ln.text) for ln in lines):
        return "Numbered sequence (GenBank style)"
    if roles.count("text") > roles.count("sequence"):
        return "Text with a sequence in it"
    return "Sequence printout"


def overlay(im: Image.Image, lines: list[Line], page: int = 1) -> Image.Image:
    out = im.copy()
    d = ImageDraw.Draw(out, "RGBA")
    w = max(2, int(max(im.size) / 600))
    for ln in lines:
        if ln.page != page:
            continue
        c = ROLE_COLORS[ln.role]
        d.rectangle(ln.box, outline=c + (255,), width=w, fill=c + (38,))
    return out


# ---------------------------------------------------------------------------------------------
def read_image(path: str) -> Reading:
    im = load_image(path)
    lines = ocr_lines(im)
    return _finish(lines, [(1, im)], "OCR", 1)


def read_pdf(path: str) -> Reading:
    import pypdfium2 as pdfium
    try:
        pdf = pdfium.PdfDocument(path)
    except Exception as e:  # noqa: BLE001
        raise Blocked("The PDF could not be opened. It may be damaged or password-protected.") from e
    n = len(pdf)
    pages = min(n, MAX_PAGES)
    text_lines, imgs, ocr_needed = [], [], False
    for i in range(pages):
        page = pdf[i]
        txt = page.get_textpage().get_text_range()
        if len(txt.strip()) >= 20:
            for t in txt.splitlines():
                if t.strip():
                    text_lines.append(Line(t.strip(), (0, 0, 0, 0), 1.0, page=i + 1))
        else:
            ocr_needed = True
            im = page.render(scale=2).to_pil().convert("RGB")
            s = MAX_SIDE / max(im.size)
            if s < 1:
                im = im.resize((int(im.width * s), int(im.height * s)))
            imgs.append((i + 1, im))
            text_lines.extend(ocr_lines(im, page=i + 1))
    reading = _finish(text_lines, imgs, "OCR" if ocr_needed else "PDF text", pages)
    if n > MAX_PAGES:
        reading.doc_type += f" (first {MAX_PAGES} of {n} pages read)"
    return reading


def _finish(lines: list[Line], images: list[Image.Image], source: str, pages: int) -> Reading:
    classify(lines)
    found = assemble(lines)
    r = Reading(doc_type(lines, found), found, lines, [], source, pages)
    r.overlays = [overlay(im, lines, page=pg) for pg, im in images]
    if not found:
        words = sum(len(ln.text.split()) for ln in lines)
        what = "no readable text" if words == 0 else f"{words} words of text but no DNA, RNA or protein sequence"
        raise Blocked(f"It has {what}. Upload an image or PDF that shows a sequence, "
                      "such as a FASTA printout, a numbered sequence or a table of sequences.")
    return r
