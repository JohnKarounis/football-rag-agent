"""Hand-written tool-calling loop on teh raw OpenAI client."""

import json
from openai import OpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

from football_rag import config
from football_rag.prompts import SYSTEM
from football_rag.tools import agent_tools

MAX_STEPS = 5

client = OpenAI(base_url=config.LMSTUDIO_URL, api_key=config.LMSTUDIO_API_KEY)
tools = [convert_to_openai_tool(t) for t in agent_tools]
available_tools = {t.name: t for t in agent_tools}

def run_agent(question, max_steps=MAX_STEPS):
    """Answer a question with a manual model -> tool -> model loop."""

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question}
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=messages,
            tools=tools,
            temperature=0
        )
        message = response.choices[0].message

        # No tool calls: the model has answered
        if not message.tool_calls:
            return message.content

        messages.append(message)
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            fn = available_tools[tool_call.function.name]
            result = fn.invoke(args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return "Stopped: reached the step limit"
