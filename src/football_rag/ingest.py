"""Ingestion: load the Wikipedia corpus, chunk it, and index it in Chroma."""

import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from football_rag import config
from football_rag.embeddings import get_embeddings


HEADERS = [("##", "h2"), ("###", "h3"), ("####", "h4")]


def load_docs():
    """Load corpus .txt files as Documents with a 'title' metadata field."""

    docs = []
    # Sorted for a deterministic chunk order
    for path in sorted(config.CORPUS_DIR.glob('*.txt')):
        text = path.read_text(encoding='utf-8')
        docs.append(Document(page_content=text, metadata={'title': path.stem}))
    return docs


def wiki_to_markdown(text):
    """Convert MediaWiki '==' headings to Markdown '#' headings."""

    new_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('==') and stripped.endswith('=='):
            level = len(stripped) - len(stripped.lstrip('='))
            name = stripped.strip('= ')
            heading_line = '#' * level + ' ' + name
            new_lines.append(heading_line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)


def split_docs(docs):
    """Split docs by heading, drop skipped sections, then chunk by word count."""

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS)
    all_sections = []
    for doc in docs:
        sections = md_splitter.split_text(wiki_to_markdown(doc.page_content))
        for section in sections:
            if section.metadata.get('h2') in config.SKIP_SECTIONS:
                continue
            section.metadata['title'] = doc.metadata['title']
            all_sections.append(section)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=lambda t: len(t.split())
    )

    return recursive_splitter.split_documents(all_sections)


def latest_year(text):
    """Return the latest year (1900-2099) mentioned in the text, or 0 if none."""

    years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', text)]
    return max(years) if years else 0 


def add_breadcrumbs(chunks):
    """Prefix each chunk with its 'title > section' path and add max_year metadata."""

    prefixed = []
    for chunk in chunks:
        metadata = chunk.metadata 
        path = ' > '.join(metadata[key] for key in ['title', 'h2', 'h3', 'h4'] if key in metadata)
        content = f'{path}: {chunk.page_content}'
        new_metadata = {**metadata, 'max_year': latest_year(chunk.page_content)}
        prefixed.append(Document(page_content=content, metadata=new_metadata))

    return prefixed

