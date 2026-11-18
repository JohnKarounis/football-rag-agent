"""Sanity checks on the committed eval sets."""

import json

from football_rag import config


def load_eval_set(name):
    """Load one eval set from data/eval by file name."""
    path = config.EVAL_DIR / f'{name}.json'
    with open(path, encoding='utf-8') as f:
        eval_set = json.load(f)
    return eval_set


def load_corpus_text():
    """Return all corpus articles joined into one string."""
    paths = sorted(config.CORPUS_DIR.glob('*.txt'))
    texts = []
    for path in paths:
        text = path.read_text(encoding='utf-8')
        texts.append(text)

    corpus = '\n'.join(texts)
    return corpus


def test_eval_set_sizes():
    """The three eval sets have 18, 30, and 11 items."""
    hand_written = load_eval_set('hand_written')
    tuning = load_eval_set('tuning')
    held_out = load_eval_set('held_out')
    assert len(hand_written) == 18
    assert len(tuning) == 30
    assert len(held_out) == 11


def test_every_item_has_the_same_keys():
    """Every item has exactly question, answer, and phrase."""
    for name in ['hand_written', 'tuning', 'held_out']:
        eval_set = load_eval_set(name)
        for item in eval_set:
            assert set(item) == {'question', 'answer', 'phrase'}


def test_every_phrase_is_in_the_corpus():
    """Every retrieval phrase appears word for word in the corpus, so retrieval metrics can find it."""
    corpus = load_corpus_text()
    for name in ['hand_written', 'tuning', 'held_out']:
        eval_set = load_eval_set(name)
        for item in eval_set:
            if item['phrase'] is not None:
                assert item['phrase'] in corpus, item['question']


def test_held_out_does_not_overlap_tuning():
    """No held-out question also appears in the tuning set."""
    tuning = load_eval_set('hand_written') + load_eval_set('tuning')
    held_out = load_eval_set('held_out')
    tuning_questions = {item['question'] for item in tuning}
    for item in held_out:
        assert item['question'] not in tuning_questions


def test_input_guardrail_has_question_and_allowed():
    """Every input guardrail item has exactly question and allowed, and allowed is a bool."""
    input_guardrail = load_eval_set('input_guardrail')
    for item in input_guardrail:
        assert set(item) == {'question', 'allowed'}
        assert isinstance(item['allowed'], bool)
    assert len(input_guardrail) == 20