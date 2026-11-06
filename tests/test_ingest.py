"""Tests for the ingestion helpers."""

from langchain_core.documents import Document

from football_rag import config
from football_rag.ingest import wiki_to_markdown, latest_year, add_breadcrumbs, load_docs, split_docs


def test_wiki_to_markdown_converts_headings():
    """== headings become ## and === become ###; other lines stay as they are."""
    text = 'Intro\n== History ==\nBody\n=== Early years ==='
    expected = 'Intro\n## History\nBody\n### Early years'
    assert wiki_to_markdown(text) == expected


def test_latest_year_returns_latest():
    """The latest of several years is returned."""
    assert latest_year('Founded in 1992, champions in 2016 and 2004.') == 2016


def test_latest_year_returns_zero_without_years():
    """Numbers that aren't 1900-2099 years are ignored, so the result is 0."""
    assert latest_year('No years here, only 12345 and 42.') == 0


def test_add_breadcrumbs_prefixes_path_and_adds_year():
    """A chunk gets a 'title > section: ' prefix and its latest year as max_year."""
    chunk = Document(page_content='Spain won in 2024.', metadata={'title': 'UEFA Euro 2024', 'h2': 'Final'})
    result = add_breadcrumbs([chunk])[0]

    assert result.page_content == 'UEFA Euro 2024 > Final: Spain won in 2024.'
    assert result.metadata['max_year'] == 2024


def test_split_docs_on_corpus():
    """The real corpus gives 565 chunks, none from skipped sections, none over CHUNK_SIZE words."""
    docs = load_docs()
    chunks = split_docs(docs)

    assert len(chunks) == 565
    for chunk in chunks:
        assert chunk.metadata.get('h2') not in config.SKIP_SECTIONS
        assert len(chunk.page_content.split()) <= config.CHUNK_SIZE