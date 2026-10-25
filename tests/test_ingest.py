"""Tests for text normalisation, chunking and id generation."""

from __future__ import annotations

import pytest

from bookrag.Rag_app.ingest import (
    chunk_text,
    make_id,
    normalise_text,
    source_key,
    split_sentences,
)


def test_normalise_collapses_whitespace():
    assert normalise_text("a  \t b\r\nc") == "a b\nc"
    assert normalise_text("") == ""


def test_normalise_collapses_blank_lines():
    assert normalise_text("a\n\n\n\nb") == "a\n\nb"


def test_split_sentences_handles_abbreviations():
    text = "See Fig. 3 for details. The end follows."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].startswith("See Fig.")


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunk_text_respects_size():
    text = "word " * 500
    chunks = chunk_text(text, size=200, overlap=0)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunk_text_produces_overlap():
    text = "Sentence number one. " * 40
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    # The tail of chunk N should reappear at the head of chunk N+1.
    tail = chunks[0][-30:]
    assert tail in chunks[1]


def test_chunk_text_keeps_paragraph_structure():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, size=1000, overlap=0)
    assert len(chunks) == 1
    assert "\n\n" in chunks[0]


def test_chunk_text_hard_splits_oversized_unit():
    text = "x" * 5000
    chunks = chunk_text(text, size=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_chunk_text_raises_on_bad_size():
    with pytest.raises(ValueError):
        chunk_text("hello", size=0)


def test_make_id_is_deterministic():
    a = make_id("text", "book.pdf", "hello", 1)
    b = make_id("text", "book.pdf", "hello", 1)
    assert a == b


def test_make_id_varies_by_page_and_content():
    base = make_id("text", "book.pdf", "hello", 1)
    assert make_id("text", "book.pdf", "hello", 2) != base
    assert make_id("text", "book.pdf", "world", 1) != base
    assert make_id("image", "book.pdf", "hello", 1) != base


def test_source_key_uses_file_name():
    assert source_key("/a/b/c.pdf") == "c.pdf"
