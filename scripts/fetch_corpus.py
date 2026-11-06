"""Download the 5 wikipedia articles into data/corpus. Overwrites the committed corpus."""

import requests

from football_rag import config

WIKI_API_URL = 'https://en.wikipedia.org/w/api.php'
USER_AGENT = 'rag-practice/1.0 (github.com/JohnKarounis)'
HTTP_AMOUNT = 10

TITLES = [
    '2022 FIFA World Cup',
    'UEFA Euro 2024',
    'FIFA World Cup',
    'Premier League',
    'Offside (association football)',
]


def fetch_article(title):
    """Return the plain-text extract of one wikipedia article."""
    params = {'action': 'query', 'prop': 'extracts', 'explaintext': 1, 'titles': title, 'format': 'json', 'redirects': 1}
    headers = {'User-Agent': USER_AGENT}
    response = requests.get(WIKI_API_URL, params=params, headers=headers, timeout=HTTP_AMOUNT)
    response.raise_for_status()

    pages = response.json()['query']['pages']
    page = list(pages.values())[0]
    text = page['extract']
    return text


if __name__ == '__main__':
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for title in TITLES:
        text = fetch_article(title)
        path = config.CORPUS_DIR / f'{title}.txt'
        path.write_text(text, encoding='utf-8')
        print(title, len(text.split()), 'words')
