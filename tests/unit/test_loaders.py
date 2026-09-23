import pytest

from app.ingestion.kubernetes import url_for
from app.ingestion.loaders import (
    DocumentParseError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    load_document,
)
from tests.helpers import make_pdf


def test_markdown_loader_uses_front_matter_title_and_preprocessor(tmp_path):
    f = tmp_path / "pods.md"
    f.write_text("---\ntitle: Pods\ndescription: About pods\n---\nHello {{x}}\n")
    doc = load_document(f, source="k/pods.md", preprocessor=lambda s: s.replace("{{x}}", "Pod"))
    assert doc.name == "Pods"
    assert doc.metadata == {"title": "Pods", "description": "About pods"}
    assert doc.text == "Hello Pod\n"
    assert doc.source_type == "markdown"


def test_text_loader_normalizes_line_endings_and_blank_runs(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_bytes(b"line one\r\n\r\n\r\n\r\nline two   \r\n")
    doc = load_document(f)
    assert doc.text == "line one\n\nline two\n"
    assert doc.name == "notes"
    assert len(doc.sections) == 1


def test_pdf_loader_records_page_sections(tmp_path):
    f = tmp_path / "manual.pdf"
    f.write_bytes(make_pdf([["Kubelet manages pods."], ["Scheduler assigns nodes."]]))
    doc = load_document(f, source="upload://manual.pdf")
    assert [s.page for s in doc.sections] == [1, 2]
    assert doc.metadata["page_count"] == 2
    page2 = doc.sections[1]
    assert "Scheduler assigns nodes." in doc.text[page2.start : page2.end]
    assert "Kubelet" not in doc.text[page2.start : page2.end]


def test_pdf_blank_pages_are_skipped_but_numbering_kept(tmp_path):
    f = tmp_path / "gap.pdf"
    f.write_bytes(make_pdf([["first"], [], ["third"]]))
    assert [s.page for s in load_document(f).sections] == [1, 3]


def test_unsupported_file_type(tmp_path):
    f = tmp_path / "slides.pptx"
    f.write_bytes(b"x")
    with pytest.raises(UnsupportedFileTypeError, match=r"\.pptx"):
        load_document(f)


def test_empty_document(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("---\ntitle: Empty\n---\n<!-- nothing -->\n")
    with pytest.raises(EmptyDocumentError):
        load_document(f)


def test_corrupt_pdf_raises_parse_error(tmp_path):
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(DocumentParseError):
        load_document(f)


def test_non_utf8_text_raises_parse_error(tmp_path):
    f = tmp_path / "latin.txt"
    f.write_bytes("caf\xe9".encode("latin-1"))
    with pytest.raises(DocumentParseError):
        load_document(f)


@pytest.mark.parametrize(
    ("source", "url"),
    [
        (
            "kubernetes/concepts/workloads/pods/pod-lifecycle.md",
            "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
        ),
        (
            "kubernetes/concepts/workloads/pods/_index.md",
            "https://kubernetes.io/docs/concepts/workloads/pods/",
        ),
        (
            "kubernetes/reference/glossary/node.md",
            "https://kubernetes.io/docs/reference/glossary/?all=true#term-node",
        ),
    ],
)
def test_kubernetes_urls(source, url):
    assert url_for(source) == url
