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


def web_search_node(state: CRAGState):
    result = search_wikipedia.invoke({'query': state['search_query']})
    attempts = state['attempts'] + 1
    return {'documents': [result], 'attempts': attempts}


def generate_node(state: CRAGState):
    context = '\n\n'.join(state['documents'])
    answer = generate_chain.invoke({'question': state['question'], 'context': context})
    return {'answer': answer}


def route_after_grade(state: CRAGState):
    if state['web_search']:
        return 'rewrite'
    return 'generate'


def route_after_generate(state: CRAGState):
    if NOT_FOUND in state['answer'] and state['attempts'] < MAX_WEB_ATTEMPTS:
        return 'rewrite'
    return END


def build_crag_graph():
    """Build the CRAG graph."""
    builder = StateGraph(CRAGState)

    builder.add_node('retrieve', retrieve_node)
    builder.add_node('grade', grade_node)
    builder.add_node('rewrite', rewrite_node)
    builder.add_node('web_search', web_search_node)
    builder.add_node('generate', generate_node)

    builder.add_edge(START, 'retrieve')
    builder.add_edge('retrieve', 'grade')
    builder.add_conditional_edges('grade', route_after_grade)
    builder.add_edge('rewrite', 'web_search')
    builder.add_edge('web_search', 'generate')
    builder.add_conditional_edges('generate', route_after_generate)

    graph = builder.compile()
    return graph


crag_graph = build_crag_graph()


def run_crag(question):
    """Run the CRAG graph on one question and return the answer."""
    result = crag_graph.invoke({'question': question, 'attempts': 0})
    return result['answer']
