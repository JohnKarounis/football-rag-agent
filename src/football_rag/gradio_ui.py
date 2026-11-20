"""Gradio chat UI that calls the football API."""

import gradio as gr
import os
import requests

API_URL = os.environ.get('API_URL', 'http://127.0.0.1:8000')
REQUEST_TIMEOUT = 120

def respond(message, history):
    """Send the user's question to the API and return the answer. Gradio passes history, but the API has no memory, so it is unused."""
    url = f'{API_URL}/v1/ask'
    response = requests.post(url, json={'question': message}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data['answer']

chatbot = gr.Chatbot(
    height=550,
    placeholder='Ask me about the World Cup, Euro 2024, or the Premier League',
    buttons=['copy']
)

demo = gr.ChatInterface(
    fn=respond, 
    title='Football Q&A', 
    description='Answers questions about the World Cup, Euro 2024, the Premier League and the offside rule, with Wikipedia as a fallback. Each question is answered on its own (no conversation memory).',
    examples=['Who won Euro 2024?', 'Who won the Premier League in 2023?', 'Who is the highest goal scorer in World Cup history?'],
    chatbot=chatbot
)


theme = gr.themes.Soft(primary_hue='green')
if __name__ == '__main__':
    demo.launch(theme=theme)