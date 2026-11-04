"""LangGraph tool-calling agent."""

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

from football_rag.llm import get_llm
from football_rag.prompts import SYSTEM
from football_rag.tools import agent_tools

RECURSION_LIMIT = 12 


def build_graph(llm, tools, system_prompt):
    """Build a model <-> tools agent graph."""

    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: MessagesState):
        prompt = [SystemMessage(content=system_prompt)] + state['messages']
        response = llm_with_tools.invoke(prompt)
        return {'messages': [response]}

    builder = StateGraph(MessagesState)
    builder.add_node('model', call_model)
    builder.add_node('tools', ToolNode(tools))

    builder.add_edge(START, 'model')
    builder.add_conditional_edges('model', tools_condition)
    builder.add_edge('tools', 'model')

    graph = builder.compile()
    return graph


graph = build_graph(get_llm(), agent_tools, SYSTEM)


def run_graph(question):
    """Run the agent graph on one question and return the final answer."""

    try:
        result = graph.invoke(
            {'messages': [HumanMessage(content=question)]},
            config={'recursion_limit': RECURSION_LIMIT}
        )
        answer = result['messages'][-1].content

    except GraphRecursionError:
        answer = "Stopped: reached the step limit"

    return answer
