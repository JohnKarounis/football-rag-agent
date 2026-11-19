"""Optional LangFuse tracing: callback handlers for LangChain, enabled by env variables."""

import os
from langfuse.langchain import CallbackHandler

def get_callbacks():
    """Return the Langfuse callback handler if both keys are set, otherwise an empty list."""
    public_key = os.environ.get('LANGFUSE_PUBLIC_KEY')
    secret_key = os.environ.get('LANGFUSE_SECRET_KEY')
    if public_key is None or secret_key is None:
        return []
    handler = CallbackHandler()
    return [handler]
    