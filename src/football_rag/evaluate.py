"""LLM-as-judge evaluation of the agents."""

import json
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import get_usage_metadata_callback

from football_rag import config
from football_rag.llm import get_llm, count_tokens
from football_rag.prompts import JUDGE_SYSTEM
from football_rag.agent_loop import run_agent
from football_rag.agent_graph import run_graph
from football_rag.crag_graph import run_crag
from football_rag.rag_chain import run_chain
from football_rag.input_guardrail import check_question
from football_rag.output_guardrail import check_answer

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
    """Run the agent on every question, time it, count its tokens and score its answers with the judge."""
    results = []
    for item in eval_set:
        question = item['question']
        start_time = time.perf_counter()
        with get_usage_metadata_callback() as usage_callback:
            answer = agent_func(question)

        latency_s = time.perf_counter() - start_time
        input_tokens, output_tokens = count_tokens(usage_callback)

        expected = item['answer']
        if expected is None:
            expected = 'Refusal'

        verdict = judge_chain.invoke({'question': question, 'expected': expected, 'answer': answer})
        judged = verdict.strip().upper() == 'CORRECT'

        results.append({'Question': question, 'Expected': expected, 'Answer': answer, 'Judge': judged, 'Latency (s)': latency_s, 'Input tokens': input_tokens, 'Output tokens': output_tokens})

    correct = sum(r['Judge'] for r in results)
    judge_rate = correct / len(results)
    avg_latency_s = sum([r['Latency (s)'] for r in results]) / len(results)
    avg_input_tokens = sum([r['Input tokens'] for r in results]) / len(results)
    avg_output_tokens = sum([r['Output tokens'] for r in results]) / len(results)
    metrics = {'Judge Rate': judge_rate, 'Avg latency (s)': round(avg_latency_s, 2), 'Avg input tokens': round(avg_input_tokens), 'Avg output tokens': round(avg_output_tokens)}
    return metrics, results


def evaluate_input_guardrail(eval_set):
    """Check the input guardrail's decision on each question against its label."""
    results = []
    for item in eval_set:
        question = item['question']
        decision = check_question(question)
        expected = item['allowed']
        predicted = decision.allowed
        correct = predicted == expected
        results.append({'Question': question, 'Expected': expected, 'Predicted': predicted, 'Correct': correct, 'Reason': decision.reason})

    num_correct = sum([r['Correct'] for r in results])
    accuracy = num_correct / len(results)
    false_blocks = 0
    missed_blocks = 0
    for result in results:
        if result['Expected'] and not result['Predicted']:
            false_blocks += 1
        if not result['Expected'] and result['Predicted']:
            missed_blocks += 1

    metrics = {'Accuracy': accuracy, 'False Blocks': false_blocks, 'Missed Blocks': missed_blocks}
    return metrics, results


def evaluate_output_guardrail(eval_set):
    """Check the output guardrail's decision on each answer against its label."""
    results = []
    for item in eval_set:
        question = item['question']
        context = item['context']
        answer = item['answer']
        decision = check_answer(question, context, answer)
        expected = item['grounded']
        predicted = decision.grounded
        reason = decision.reason
        correct = expected == predicted
        results.append({'Question': question, 'Answer': answer, 'Expected': expected, 'Predicted': predicted, 'Correct': correct, 'Reason': reason})

    num_correct = sum([r['Correct'] for r in results])
    accuracy = num_correct / len(results)
    false_rejects = 0
    missed_hallucinations = 0
    for result in results:
        if result['Expected'] and not result['Predicted']:
            false_rejects += 1 
        if not result['Expected'] and result['Predicted']:
            missed_hallucinations += 1

    metrics = {'Accuracy': accuracy, 'False Rejects': false_rejects, 'Missed Hallucinations': missed_hallucinations}
    return metrics, results


if __name__ == '__main__':
    tuning_set = load_eval_set('hand_written') + load_eval_set('tuning')
    held_out = load_eval_set('held_out')
    input_guardrail_set = load_eval_set('input_guardrail')
    output_guardrail_set = load_eval_set('output_guardrail')

    input_guardrail_metrics, _ = evaluate_input_guardrail(input_guardrail_set)
    print('Input Guardrail:', input_guardrail_metrics)

    output_guardrail_metrics, _ = evaluate_output_guardrail(output_guardrail_set)
    print('Output guardrail:', output_guardrail_metrics)

    agents = {'RAG chain': run_chain, 'Agent loop': run_agent, 'LangGraph agent': run_graph, 'CRAG': run_crag}
    for name, agent_func in agents.items():
        tuning_metrics, _ = evaluate_agent(agent_func, tuning_set)
        held_out_metrics, _ = evaluate_agent(agent_func, held_out)
        
        print(name, '| Tuning:', tuning_metrics, '| Held-out:', held_out_metrics)
