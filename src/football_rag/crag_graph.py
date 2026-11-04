"""Corrective RAG graph: retrieve, grade, fall back to Wikipedia, generate."""

from typing import TypedDict

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from football_rag.llm import get_llm
from football_rag.prompts import GRADER_SYSTEM, REWRITE_SYSTEM, GENERATE_SYSTEM
from football_rag.tools import retriever, search_wikipedia

MAX_WEB_ATTEMPTS = 2
NOT_FOUND = "couldn't find that"

llm = get_llm()


class CRAGState(TypedDict):
    question: str
    documents: list[str]
    web_search: bool
    search_query: str
    answer: str
    attempts: int


class GradeChunk(BaseModel):
    """Relevance grade for a retrieved chunk"""

    relevant: bool = Field(description="True if the chunk contains information that helps answer the question, otherwise False")


grader_prompt = ChatPromptTemplate.from_messages([
    ('system', GRADER_SYSTEM),
    ('human', 'Question:\n{question}\n\nDocument:\n{document}')
])
grader_chain = grader_prompt | llm.with_structured_output(GradeChunk)

rewrite_prompt = ChatPromptTemplate.from_messages([
    ('system', REWRITE_SYSTEM),
    ('human', 'Question:\n{question}\n\nPrevious query:\n{previous_query}')
])
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

generate_prompt = ChatPromptTemplate.from_messages([
    ('system', GENERATE_SYSTEM),
    ('human', 'Question:\n{question}\n\nContext:\n{context}')
])
generate_chain = generate_prompt | llm | StrOutputParser()


def retrieve_node(state: CRAGState):
    docs = retriever.invoke(state['question'])
    documents = [doc.page_content for doc in docs]
    return {'documents': documents}


def grade_node(state: CRAGState):
    relevant_chunks = []
    for doc in state['documents']:
        grade = grader_chain.invoke({'question': state['question'], 'document': doc})
        if grade.relevant:
            relevant_chunks.append(doc)

    web_search = len(relevant_chunks) == 0
    return {'documents': relevant_chunks, 'web_search': web_search}


def rewrite_node(state: CRAGState):
    previous_query = state.get('search_query', 'none')
    search_query = rewrite_chain.invoke({'question': state['question'], 'previous_query': previous_query})
    return {'search_query': search_query}