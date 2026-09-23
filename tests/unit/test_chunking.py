from itertools import pairwise
from pathlib import Path

import pytest

from app.config.settings import ChunkingSettings
from app.ingestion.chunking import StructureAwareChunker, section_at
from app.ingestion.loaders import load_markdown
from app.ingestion.tokens import estimate_tokens
from app.ingestion.types import LoadedDocument, Section

CORPUS = Path("data/raw/kubernetes-website")


def chunker(child=60, overlap=10, parent=200) -> StructureAwareChunker:
    return StructureAwareChunker(
        ChunkingSettings(
            child_chunk_size=child, child_chunk_overlap=overlap, parent_chunk_size=parent
        )
    )


def paragraph(topic: str, sentences: int = 4) -> str:
    return " ".join(
        f"The {topic} component handles case {i} with several distinct words."
        for i in range(sentences)
    )


def sample_doc() -> LoadedDocument:
    parts = ["# Guide", paragraph("intro", 2)]
    for sec in ("Scheduling", "Networking", "Storage"):
        parts += [f"## {sec}", paragraph(sec.lower()), paragraph(sec.lower() + "-b")]
        parts += [f"### {sec} details", paragraph(sec.lower() + "-c", 6)]
    parts += ["## Example", "```yaml\napiVersion: v1\nkind: Pod\n\nmetadata:\n  name: x\n```"]
    return load_markdown("\n\n".join(parts) + "\n", source="test://guide.md")


def assert_invariants(doc: LoadedDocument, ch: StructureAwareChunker) -> None:
    parents, children = ch.chunk(doc)
    s = ch.settings
    text = doc.text
    assert parents and children
    for p in parents:
        assert 0 <= p.start < p.end <= len(text)
        assert p.tokens == estimate_tokens(text[p.start : p.end])
        assert p.tokens <= s.parent_chunk_size
    for c in children:
        p = parents[c.parent_index]
        assert p.start <= c.start < c.end <= p.end, "child escapes its parent"
        assert c.tokens <= s.child_chunk_size
        assert not text[c.start].isspace() and not text[c.end - 1].isspace()

    def covered(spans, length):
        mask = [False] * length
        for a, b in spans:
            for i in range(a, b):
                mask[i] = True
        return mask

    # No content is lost: every non-whitespace character is in some parent and some child.
    for spans in ([(p.start, p.end) for p in parents], [(c.start, c.end) for c in children]):
        mask = covered(spans, len(text))
        lost = [i for i, ch_ in enumerate(text) if not ch_.isspace() and not mask[i]]
        assert not lost, f"lost text: {text[lost[0] : lost[0] + 40]!r}"


def test_invariants_on_synthetic_document():
    assert_invariants(sample_doc(), chunker())


def test_chunking_is_deterministic():
    doc = sample_doc()
    assert chunker().chunk(doc) == chunker().chunk(doc)


def test_code_fence_is_never_split_when_it_fits():
    doc = sample_doc()
    _, children = chunker().chunk(doc)
    fence_start = doc.text.index("```yaml")
    fence_end = doc.text.index("```", fence_start + 3) + 3
    holders = [c for c in children if c.start <= fence_start and c.end >= fence_end]
    assert holders, "code block split across chunks"


def test_heading_is_glued_to_following_content():
    doc = sample_doc()
    _, children = chunker().chunk(doc)
    for c in children:
        chunk_text = doc.text[c.start : c.end].strip()
        assert not (chunk_text.startswith("#") and "\n" not in chunk_text), chunk_text


def test_overlap_repeats_trailing_context_and_zero_overlap_disables_it():
    doc = sample_doc()
    _, with_overlap = chunker(overlap=20).chunk(doc)
    pairs = [(a, b) for a, b in pairwise(with_overlap) if a.parent_index == b.parent_index]
    assert any(b.start < a.end for a, b in pairs)
    _, no_overlap = chunker(overlap=0).chunk(doc)
    for a, b in pairwise(no_overlap):
        assert b.start >= a.end


def test_oversized_paragraph_and_giant_token_respect_size_limits():
    long_para = paragraph("huge", 30)
    giant_word = "x" * 5000  # e.g. an inline base64 blob: no whitespace to split on
    doc = load_markdown(f"## A\n\n{long_para}\n\n{giant_word}\n", source="test://big.md")
    ch = chunker(child=50, overlap=0, parent=100)
    _, children = ch.chunk(doc)
    assert len(children) > 5
    assert_invariants(doc, ch)  # includes: no child > 50 tokens, no text lost
    giant = [c for c in children if set(doc.text[c.start : c.end]) == {"x"}]
    assert sum(c.end - c.start for c in giant) == len(giant_word)


def test_estimator_costs_long_mixed_and_camelcase_runs():
    assert estimate_tokens("pod") == 1
    assert estimate_tokens("a" * 22) == 4  # 1 token per ~7 chars
    assert estimate_tokens("4dccb216c4adb") == 7  # hex IDs split finely in WordPiece
    assert estimate_tokens("PodDisruptionBudget") == 4  # Pod + Disruption(2) + Budget
    assert estimate_tokens("kube-apiserver --v=2") == 9  # "apiserver" > 7 chars: 2 tokens


def test_pdf_pages_are_never_merged_into_one_parent():
    text = "page one text.\n\npage two text.\n"
    doc = LoadedDocument(
        source="test://x.pdf",
        name="x",
        source_type="pdf",
        text=text,
        sections=[Section(0, 14, page=1), Section(16, 30, page=2)],
    )
    parents, _ = chunker().chunk(doc)
    assert [p.page for p in parents] == [1, 2]


def test_fingerprint_changes_with_config():
    assert chunker(child=60).fingerprint() != chunker(child=70).fingerprint()
    assert chunker().fingerprint() == chunker().fingerprint()


def test_section_at_returns_most_specific_section():
    doc = sample_doc()
    offset = doc.text.index("scheduling-c component")
    assert section_at(doc.sections, offset).label == "Guide > Scheduling > Scheduling details"


@pytest.mark.slow
@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not fetched")
def test_invariants_hold_on_real_kubernetes_corpus_sample():
    from app.ingestion.kubernetes import KubernetesCorpus
    from app.ingestion.loaders import EmptyDocumentError, load_document

    corpus = KubernetesCorpus(CORPUS)
    resolver = corpus.resolver()
    ch = StructureAwareChunker(ChunkingSettings())
    for path, source in corpus.files()[::15]:
        try:
            doc = load_document(path, source=source, preprocessor=resolver)
        except EmptyDocumentError:
            continue
        assert "{{<" not in doc.text and "{{%" not in doc.text, source
        assert_invariants(doc, ch)


def test_hard_wrapped_prose_is_not_split_mid_sentence():
    # Mirrors the kubernetes/website style: one sentence wrapped across several lines.
    sentence = (
        "If the liveness probe fails, the kubelet\n"
        "kills the container and the container\n"
        "is subjected to its restart policy."
    )
    doc = load_markdown("## A\n\n" + " ".join([sentence] * 12) + "\n", source="test://wrap.md")
    _, children = chunker(child=60, overlap=20, parent=400).chunk(doc)
    assert len(children) > 2
    for c in children[1:]:
        assert doc.text[c.start : c.end].startswith("If the liveness"), doc.text[
            c.start : c.start + 30
        ]
