"""Gradio chat UI that calls the football API."""

import gradio as gr
import os
import requests

API_URL = os.environ.get('API_URL', 'http://127.0.0.1:8000')
REQUEST_TIMEOUT = 120

def respond(message, history, conversation_id):
    """Send the user's question with the conversation id to the API; return the answer and the id to keep. An empty history means a new chat, so a new conversation."""
    if not history:
        conversation_id = None
    url = f'{API_URL}/v1/ask'
    response = requests.post(url, json={'question': message, 'conversation_id': conversation_id}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data['answer'], data['conversation_id']

chatbot = gr.Chatbot(
    height=550,
    placeholder='Ask me about the World Cup, Euro 2024, or the Premier League',
    buttons=['copy']
)

conversation_state = gr.State(None)
examples = [
    ['Who won Euro 2024?', None], 
    ['Who won the Premier League in 2023?', None], 
    ['Who is the highest goal scorer in World Cup history?', None]
]

demo = gr.ChatInterface(
    fn=respond, 
    title='Football Q&A', 
    description='Answers questions about the World Cup, Euro 2024, the Premier League and the offside rule, with Wikipedia as a fallback. It remembers earlier questions in the chat, so you can ask follow-ups.',
    examples=examples,
    chatbot=chatbot,
    additional_inputs=[conversation_state],
    additional_outputs=[conversation_state]
)


theme = gr.themes.Soft(primary_hue='green')
if __name__ == '__main__':
    demo.launch(theme=theme, inbrowser=True, server_name='127.0.0.1')