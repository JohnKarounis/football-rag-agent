"""Fixed RAG chain (LCEL): retrieve once, then answer from the retrieved chunks."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from football_rag.llm import get_llm
from football_rag.prompts import RAG_SYSTEM
from football_rag.tools import retriever, format_docs

rag_prompt = ChatPromptTemplate.from_messages([
    ('system', RAG_SYSTEM),
    ('human', 'Context:\n{context}\n\nQuestion:\n{question}')
])

rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | rag_prompt
    | get_llm()
    | StrOutputParser()
)


def run_chain(question):
    """Run the RAG chain on one question and return the answer."""
    answer = rag_chain.invoke(question)
    return answer
