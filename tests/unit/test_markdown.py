from itertools import pairwise

from app.ingestion.markdown import (
    markdown_sections,
    normalize_markdown,
    parse_heading,
    split_front_matter,
)


def test_front_matter_is_parsed_and_removed():
    meta, body = split_front_matter("---\ntitle: Pod Lifecycle\nweight: 30\n---\n\nBody text\n")
    assert meta == {"title": "Pod Lifecycle", "weight": 30}
    assert body.strip() == "Body text"


def test_malformed_front_matter_does_not_crash():
    meta, body = split_front_matter("---\ntitle: [unclosed\n---\nBody\n")
    assert meta == {}
    assert "Body" in body


def test_no_front_matter():
    assert split_front_matter("# Title\n") == ({}, "# Title\n")


def test_normalize_strips_comments_anchors_and_blank_runs():
    raw = "<!-- overview -->\n## Pod phase {#pod-phase}\n\n\n\nText   \r\n"
    assert normalize_markdown(raw) == "## Pod phase\n\nText\n"


def test_parse_heading_keeps_trailing_hash_in_words():
    assert parse_heading("## C# bindings") == (2, "C# bindings")
    assert parse_heading("## Closed ##") == (2, "Closed")
    assert parse_heading("#hashtag") is None


def test_sections_carry_heading_path_and_ignore_code_fences():
    text = normalize_markdown(
        "Intro.\n\n## A\n\na text\n\n```bash\n# not a heading\n```\n\n"
        "### A1\n\na1 text\n\n## B\n\nb text\n"
    )
    sections = markdown_sections(text)
    assert [s.heading_path for s in sections] == [(), ("A",), ("A", "A1"), ("B",)]
    assert sections[2].label == "A > A1"
    # Sections tile the text exactly.
    assert sections[0].start == 0 and sections[-1].end == len(text)
    for prev, nxt in pairwise(sections):
        assert prev.end == nxt.start
    assert "# not a heading" in text[sections[1].start : sections[1].end]


def test_tilde_fences_are_respected():
    text = "## A\n\n~~~\n## fake\n~~~\n\n## B\n\nx\n"
    assert [s.heading_path for s in markdown_sections(text)] == [("A",), ("B",)]
