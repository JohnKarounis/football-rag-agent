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
