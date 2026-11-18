"""LangGraph tool-calling agent."""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

from football_rag.llm import get_llm, count_tokens
from football_rag.prompts import SYSTEM
from football_rag.tools import agent_tools
from football_rag.input_guardrail import check_question

RECURSION_LIMIT = 12
REFUSAL = 'I can only help with football.'
NOT_FOUND = "I couldn't find that."

def guardrail_node(state: MessagesState):
    """Block off-topic or injection questions with a refusal; let allowed ones through unchanged."""
    question = state['messages'][-1].content
    decision = check_question(question)
    if not decision.allowed:
        return {'messages': [AIMessage(content=REFUSAL)]}
    return {}


def route_after_guardrail(state: MessagesState):
    """Go to END if the guardrail added a refusal, otherwise to the model."""
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage):
        return END
    return 'model'


def collect_tool_context(messages):
    """Join the content of every tool result in the messages into one context string."""
    results = []
    for message in messages:
        if isinstance(message, ToolMessage):
            results.append(message.content)
    context = '\n\n'.join(results)
    return context


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
    builder.add_node('guardrail', guardrail_node)

    builder.add_edge(START, 'guardrail')
    builder.add_conditional_edges('model', tools_condition)
    builder.add_edge('tools', 'model')
    builder.add_conditional_edges('guardrail', route_after_guardrail, ['model', END])

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


def run_graph_with_usage(question):
    """Run the agent graph on one question and return the answer with its token usage."""
    with get_usage_metadata_callback() as usage_callback:
        answer = run_graph(question)

    input_tokens, output_tokens = count_tokens(usage_callback)
    return answer, input_tokens, output_tokens
