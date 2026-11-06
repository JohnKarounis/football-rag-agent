"""LLM-as-judge evaluation of the agents."""

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from football_rag import config
from football_rag.llm import get_llm
from football_rag.prompts import JUDGE_SYSTEM
from football_rag.agent_loop import run_agent
from football_rag.agent_graph import run_graph
from football_rag.crag_graph import run_crag
from football_rag.rag_chain import run_chain

judge_prompt = ChatPromptTemplate.from_messages([
    ('system', JUDGE_SYSTEM),
    ('human', 'Question:\n{question}\n\nReference answer:\n{expected}\n\nModel answer:\n{answer}')
])
judge_chain = judge_prompt | get_llm() | StrOutputParser()


def load_eval_set(name):
    """Load one eval set from data/eval by file name."""
    path = config.EVAL_DIR / f'{name}.json'
    with open(path, encoding='utf-8') as f:
        eval_set = json.load(f)
    return eval_set


def evaluate_agent(agent_func, eval_set):
    """Run the agent on every question and score its answers with the judge."""
    results = []
    for item in eval_set:
        question = item['question']
        answer = agent_func(question)

        expected = item['answer']
        if expected is None:
            expected = 'Refusal'

        verdict = judge_chain.invoke({'question': question, 'expected': expected, 'answer': answer})
        judged = verdict.strip().upper() == 'CORRECT'

        results.append({'Question': question, 'Expected': expected, 'Answer': answer, 'Judge': judged})

    correct = sum(r['Judge'] for r in results)
    judge_rate = correct / len(results)
    return {'Judge Rate': judge_rate}, results


if __name__ == '__main__':
    tuning_set = load_eval_set('hand_written') + load_eval_set('tuning')
    held_out = load_eval_set('held_out')

    agents = {'RAG chain': run_chain, 'Agent loop': run_agent, 'LangGraph agent': run_graph, 'CRAG': run_crag}
    for name, agent_func in agents.items():
        tuning_metrics, _ = evaluate_agent(agent_func, tuning_set)
        held_out_metrics, _ = evaluate_agent(agent_func, held_out)
        print(name, '| tuning:', tuning_metrics, '| held-out:', held_out_metrics)
