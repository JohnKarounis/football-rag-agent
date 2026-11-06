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
    assert len(load_eval_set('hand_written')) == 18
    assert len(load_eval_set('tuning')) == 30
    assert len(load_eval_set('held_out')) == 11


def test_every_item_has_the_same_keys():
    """Every item has exactly question, answer, and phrase."""
    for name in ['hand_written', 'tuning', 'held_out']:
        for item in load_eval_set(name):
            assert set(item) == {'question', 'answer', 'phrase'}
