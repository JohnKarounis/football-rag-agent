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


def test_every_phrase_is_in_the_corpus():
    """Every retrieval phrase appears word for word in the corpus, so retrieval metrics can find it."""
    corpus = load_corpus_text()
    for name in {'hand_written', 'tuning', 'held_out'}:
        for item in load_eval_set(name):
            if item['phrase'] is not None:
                assert item['phrase'] in corpus, item['question']


def test_held_out_does_not_overlap_tuning():
    """No held-out question also appears in the tuning set."""
    tuning = load_eval_set('hand_written') + load_eval_set('tuning')
    tuning_questions = {item['question'] for item in tuning}
    for item in load_eval_set('held_out'):
        assert item['question'] not in tuning_questions
