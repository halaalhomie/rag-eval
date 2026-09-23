"""Test helpers."""

from __future__ import annotations


def make_pdf(pages: list[list[str]]) -> bytes:
    """Build a minimal valid PDF: one text line per list item, one page per inner list.

    Avoids a PDF-writing dependency just for tests. Page text is Helvetica, so pypdf's
    extraction round-trips ASCII reliably.
    """
    n = len(pages)
    font_id = 3 + 2 * n
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{' '.join(f'{3 + 2 * i} 0 R' for i in range(n))}] "
            f"/Count {n} >>"
        ).encode(),
    ]
    for i, lines in enumerate(pages):
        ops = ["BT", "/F1 12 Tf", "72 720 Td"]
        for j, line in enumerate(lines):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"{'0 -16 Td ' if j else ''}({escaped}) Tj")
        ops.append("ET")
        stream = "\n".join(ops).encode()
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + 2 * i} 0 R >>"
            ).encode()
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


class KeywordEmbedder:
    """Deterministic embedder with predictable similarity: one dimension per keyword.

    Texts sharing keywords are close; the last dimension is a small constant so no vector is
    all-zero (cosine distance is undefined for zero vectors).
    """

    VOCAB = ("probe", "deployment", "secret", "volume", "network", "scheduler", "rollback")

    def __init__(self, dim: int = 384, model_name: str = "keyword-embedder"):
        self.dim = dim
        self.model_name = model_name
        self.document_calls = 0

    def _vec(self, text: str) -> list[float]:
        lower = text.lower()
        v = [0.0] * self.dim
        for i, word in enumerate(self.VOCAB):
            v[i] = float(lower.count(word))
        v[-1] = 0.1
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v]

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)
