from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

# Local imports 
from football_rag import config

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# Create a NomicEmbeddings class
class NomicEmbeddings(Embeddings):
    """Nomic embeddings with the task prefixes nomic-embed-text requires.

    Adds 'search_document: ' to texts being indexed and 'search_query: ' to queries.
    """

    def __init__(self, base):
        self.base = base

    def embed_documents(self, texts):
        return self.base.embed_documents([DOC_PREFIX + t for t in texts])

    def embed_query(self, text):
        return self.base.embed_query(QUERY_PREFIX + text)


# Create a func that builds the client, talks to LM and wraps it in NomicEmbeddings
def get_embeddings():
    base = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.LMSTUDIO_URL,
        api_key=config.LMSTUDIO_API_KEY,
        check_embedding_ctx_length=False
    )

    return NomicEmbeddings(base)
