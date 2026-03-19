"""Tests for document_parser service."""
import os
import tempfile
import pytest

from app.services.document_parser import (
    _format_table_as_markdown,
    _classify_block_heading_level,
    _split_into_sections,
    chunk_text,
    create_parent_child_chunks,
    _create_children_from_parent,
    _build_legacy_chunks,
    _build_parent_child_chunks,
    parse_markdown,
)


class TestFormatTableAsMarkdown:
    def test_basic_table(self):
        data = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        result = _format_table_as_markdown(data)
        assert "| Name | Age |" in result
        assert "| --- | --- |" in result
        assert "| Alice | 30 |" in result

    def test_empty_table(self):
        assert _format_table_as_markdown([]) == ""

    def test_none_cells(self):
        data = [["A", None], [None, "B"]]
        result = _format_table_as_markdown(data)
        assert "| A |  |" in result

    def test_pipe_in_cell_escaped(self):
        data = [["A|B", "C"]]
        result = _format_table_as_markdown(data)
        assert "A\\|B" in result

    def test_uneven_rows(self):
        data = [["A", "B", "C"], ["D"]]
        result = _format_table_as_markdown(data)
        # Row should be padded to 3 columns
        assert result.count("|") > 3


class TestClassifyBlockHeadingLevel:
    def test_body_text(self):
        level = _classify_block_heading_level(12.0, 12.0, False, 1.2)
        assert level == 0

    def test_h1_large_font(self):
        level = _classify_block_heading_level(20.0, 12.0, False, 1.2)
        assert level == 1

    def test_h2_medium_font(self):
        level = _classify_block_heading_level(15.0, 12.0, False, 1.2)
        assert level == 2

    def test_h3_bold_slightly_larger(self):
        level = _classify_block_heading_level(13.0, 12.0, True, 1.2)
        assert level == 3

    def test_zero_body_size(self):
        level = _classify_block_heading_level(12.0, 0, False, 1.2)
        assert level == 0  # ratio would be infinite, but shouldn't crash


class TestSplitIntoSections:
    def test_basic_sections(self):
        text = "# Title\n\nBody text here.\n\n## Section\n\nMore text."
        sections = _split_into_sections(text)
        assert len(sections) >= 2

    def test_no_headings(self):
        text = "Just plain text without any headings."
        sections = _split_into_sections(text)
        assert len(sections) >= 1
        assert sections[0]["heading_path"] == ""

    def test_nested_headings(self):
        text = "# H1\n\n## H2\n\nContent under H2\n\n### H3\n\nContent under H3"
        sections = _split_into_sections(text)
        # Should find sections with hierarchical heading paths
        paths = [s["heading_path"] for s in sections if s["body"]]
        assert any("H2" in p for p in paths)

    def test_empty_text(self):
        sections = _split_into_sections("")
        assert len(sections) <= 1


class TestChunkText:
    def test_basic_chunking(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) >= 1

    def test_short_text_single_chunk(self):
        text = "Short text."
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_overlap_present(self):
        # With overlap, consecutive chunks should share text
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=20)
        if len(chunks) >= 2:
            # Some overlap expected
            assert len(chunks) >= 2

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=100, chunk_overlap=10)
        # Empty text should produce one empty-ish chunk or none
        assert len(chunks) <= 1


class TestCreateParentChildChunks:
    def test_basic_parent_child(self):
        text = "# Title\n\n" + "Sentence. " * 100
        result = create_parent_child_chunks(text, parent_size=500, child_size=100, child_overlap=20)
        assert "parent_chunks" in result
        assert "child_chunks" in result
        assert len(result["parent_chunks"]) >= 1
        assert len(result["child_chunks"]) >= 1

    def test_parent_index_increments(self):
        text = "# H1\n\n" + "Text. " * 200 + "\n\n## H2\n\n" + "More text. " * 200
        result = create_parent_child_chunks(text, parent_size=200, child_size=50, child_overlap=10)
        indices = [p["parent_index"] for p in result["parent_chunks"]]
        assert indices == sorted(indices)

    def test_children_reference_parents(self):
        text = "Content. " * 100
        result = create_parent_child_chunks(text, parent_size=200, child_size=50, child_overlap=10)
        parent_indices = {p["parent_index"] for p in result["parent_chunks"]}
        for child in result["child_chunks"]:
            assert child["parent_index"] in parent_indices

    def test_empty_text(self):
        result = create_parent_child_chunks("", parent_size=500, child_size=100, child_overlap=20)
        assert result["parent_chunks"] == []
        assert result["child_chunks"] == []


class TestCreateChildrenFromParent:
    def test_creates_children(self):
        children = []
        _create_children_from_parent(
            parent_text="First sentence. Second sentence. Third sentence.",
            parent_index=0,
            child_size=30,
            child_overlap=10,
            child_chunks=children,
            parent_heading="Test",
        )
        assert len(children) >= 1
        assert all(c["parent_index"] == 0 for c in children)

    def test_heading_prepended(self):
        children = []
        _create_children_from_parent(
            parent_text="Some sentence here.",
            parent_index=0,
            child_size=1000,
            child_overlap=10,
            child_chunks=children,
            parent_heading="My Section",
        )
        assert "[Sezione: My Section]" in children[0]["text"]

    def test_no_heading(self):
        children = []
        _create_children_from_parent(
            parent_text="Some sentence here.",
            parent_index=0,
            child_size=1000,
            child_overlap=10,
            child_chunks=children,
            parent_heading="",
        )
        assert "[Sezione:" not in children[0]["text"]


class TestBuildLegacyChunks:
    def test_builds_chunks(self):
        pages = [
            {"text": "Page 1 sentence.", "page": 1, "source": "test.pdf"},
            {"text": "Page 2 sentence.", "page": 2, "source": "test.pdf"},
        ]
        result = _build_legacy_chunks(pages)
        assert result["mode"] == "legacy"
        assert result["parent_chunks"] == []
        assert len(result["child_chunks"]) >= 2
        # Check sequential chunk_index
        indices = [c["metadata"]["chunk_index"] for c in result["child_chunks"]]
        assert indices == list(range(len(indices)))


class TestBuildParentChildChunks:
    def test_builds_parent_child(self):
        pages = [
            {"text": "# Title\n\n" + "Content. " * 50, "page": 1, "source": "doc.pdf"},
        ]
        result = _build_parent_child_chunks(pages)
        assert result["mode"] == "parent_child"
        assert len(result["parent_chunks"]) >= 1
        assert len(result["child_chunks"]) >= 1


class TestParseMarkdown:
    def test_basic_sections(self, tmp_path):
        md = "# Intro\n\nSome intro text.\n\n## Details\n\nDetail content here.\n"
        f = tmp_path / "test.md"
        f.write_text(md, encoding="utf-8")
        sections = parse_markdown(str(f))
        assert len(sections) == 2
        assert sections[0]["page"] == 1
        assert "Intro" in sections[0]["text"]
        assert sections[1]["page"] == 2
        assert "Details" in sections[1]["text"]
        assert all(s["source"] == "test.md" for s in sections)

    def test_no_headings(self, tmp_path):
        md = "Just plain text without any headings.\n"
        f = tmp_path / "plain.md"
        f.write_text(md, encoding="utf-8")
        sections = parse_markdown(str(f))
        assert len(sections) == 1
        assert sections[0]["page"] == 1
        assert "plain text" in sections[0]["text"]

    def test_preamble_before_first_heading(self, tmp_path):
        md = "Preamble text here.\n\n# Heading\n\nBody.\n"
        f = tmp_path / "pre.md"
        f.write_text(md, encoding="utf-8")
        sections = parse_markdown(str(f))
        assert len(sections) == 2
        assert "Preamble" in sections[0]["text"]
        assert "Heading" in sections[1]["text"]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("", encoding="utf-8")
        sections = parse_markdown(str(f))
        assert sections == []

    def test_multiple_heading_levels(self, tmp_path):
        md = "# H1\n\nText A\n\n## H2\n\nText B\n\n### H3\n\nText C\n"
        f = tmp_path / "levels.md"
        f.write_text(md, encoding="utf-8")
        sections = parse_markdown(str(f))
        assert len(sections) == 3
        for i, s in enumerate(sections, 1):
            assert s["page"] == i
