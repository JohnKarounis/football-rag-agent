"""LangGraph tool-calling agent."""

import uuid

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.memory import MemorySaver

from football_rag.llm import get_llm, count_tokens
from football_rag.prompts import SYSTEM
from football_rag.tools import agent_tools
from football_rag.input_guardrail import check_question
from football_rag.output_guardrail import check_answer
from football_rag.tracing import get_callbacks

RECURSION_LIMIT = 12
REFUSAL = 'I can only help with football.'
NOT_FOUND = "I couldn't find that."
RETRY_FEEDBACK = "Your answer is not supported by the search results."
MAX_ANSWER_RETRIES = 1

def input_guardrail_node(state: MessagesState):
    """Block off-topic or injection questions with a refusal; let allowed ones through unchanged."""
    question = state['messages'][-1].content
    decision = check_question(question)
    if not decision.allowed:
        return {'messages': [AIMessage(content=REFUSAL)]}
    return {}


def route_after_input_guardrail(state: MessagesState):
    """Go to END if the input guardrail added a refusal, otherwise to the model."""
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


def count_retries(messages):
    """Count how many times the output guardrail has already sent feedback for a rewrite."""
    counter = 0
    for message in messages:
        if isinstance(message, HumanMessage) and message.content.startswith(RETRY_FEEDBACK):
            counter += 1
    return counter


def find_question_index(messages):
    """Return the position of the user's current question: the last human message that is not guardrail feedback."""
    question_index = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage) and not message.content.startswith(RETRY_FEEDBACK):
            question_index = index
    return question_index


def output_guardrail_node(state: MessagesState):
    """Check the answer against the tool results; ask for one rewrite if it is not grounded, then give up with NOT_FOUND."""
    messages = state['messages']
    question_index = find_question_index(messages)
    question = messages[question_index].content
    answer = messages[-1].content
    turn_messages = messages[question_index:]

    if answer == REFUSAL or answer == NOT_FOUND:
        return {}
    context = collect_tool_context(messages)
    decision = check_answer(question, context, answer)

    if not decision.grounded:
        retries = count_retries(turn_messages)
        if retries < MAX_ANSWER_RETRIES:
            feedback = f"{RETRY_FEEDBACK} {decision.reason} Rewrite it using only facts from the search results, or reply exactly: '{NOT_FOUND}'"
            return {'messages': [HumanMessage(content=feedback)]}
        return {'messages': [AIMessage(content=NOT_FOUND)]}
    return {}


def route_after_output_guardrail(state: MessagesState):
    """Go back to the model if the output guardrail asked for a rewrite, otherwise to END."""
    last_message = state['messages'][-1]
    if isinstance(last_message, HumanMessage):
        return 'model'
    return END


def build_graph(llm, tools, system_prompt):
    """Build a model <-> tools agent graph."""

    llm_with_tools = llm.bind_tools(tools)
    checkpointer = MemorySaver()

    def call_model(state: MessagesState):
        prompt = [SystemMessage(content=system_prompt)] + state['messages']
        response = llm_with_tools.invoke(prompt)
        return {'messages': [response]}

    builder = StateGraph(MessagesState)
    builder.add_node('model', call_model)
    builder.add_node('tools', ToolNode(tools))
    builder.add_node('input_guardrail', input_guardrail_node)
    builder.add_node('output_guardrail', output_guardrail_node)

    builder.add_edge(START, 'input_guardrail')
    builder.add_conditional_edges('model', tools_condition, {'tools': 'tools', END: 'output_guardrail'})
    builder.add_edge('tools', 'model')
    builder.add_conditional_edges('input_guardrail', route_after_input_guardrail, ['model', END])
    builder.add_conditional_edges('output_guardrail', route_after_output_guardrail, ['model', END])

    graph = builder.compile(checkpointer=checkpointer)
    return graph


graph = build_graph(get_llm(), agent_tools, SYSTEM)


def run_graph(question, thread_id=None):
    """Run the agent graph on one question in a conversation (thread_id; a new one if missing) and return the final answer."""
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    callbacks = get_callbacks()
    try:
        result = graph.invoke(
            {'messages': [HumanMessage(content=question)]},
            config={'recursion_limit': RECURSION_LIMIT, 
                    'callbacks': callbacks, 
                    'configurable': {'thread_id': thread_id}}
        )
        answer = result['messages'][-1].content
    except GraphRecursionError:
        answer = NOT_FOUND
    return answer


def run_graph_with_usage(question, thread_id=None):
    """Run the agent graph on one question in a conversation (thread_id; a new one if missing) and return the answer with its token usage."""
    with get_usage_metadata_callback() as usage_callback:
        answer = run_graph(question, thread_id)

    input_tokens, output_tokens = count_tokens(usage_callback)
    return answer, input_tokens, output_tokens
