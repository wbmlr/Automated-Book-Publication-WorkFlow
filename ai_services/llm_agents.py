# In ai_services/llm_agents.py

import os
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping
import requests
import json

from config import GEMINI_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY

MODEL_MAP = {
    "gemini": "gemini-1.5-flash-latest",
    "groq": "llama3-8b-8192",
    "cerebras": "llama-4-scout-17b-16e-instruct"
}

class CustomChatCerebras(LLM):
    model_name: str = MODEL_MAP["cerebras"]
    @property
    def _llm_type(self) -> str: return "custom_cerebras"
    def _call(self, prompt: str, **kwargs: Any) -> str:
        response = requests.post(
            url="https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
            data=json.dumps({"model": self.model_name, "messages": [{"role": "user", "content": prompt}]})
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    @property
    def _identifying_params(self) -> Mapping[str, Any]: return {"model_name": self.model_name}

# --- LLM Factory Function ---
def get_llm_chain(provider: str):
    """
    This function now ONLY creates and returns the LLM instance,
    with streaming explicitly enabled where supported.
    """
    model_name = MODEL_MAP.get(provider)
    if not model_name:
        raise ValueError(f"Invalid provider specified: {provider}")

    if provider == "gemini":
        # FIX: Explicitly enable token-by-token streaming from the provider.
        return ChatGoogleGenerativeAI(model=model_name, api_key=GEMINI_API_KEY, streaming=True)
    
    if provider == "groq":
        # FIX: Explicitly enable token-by-token streaming from the provider.
        return ChatGroq(model_name=model_name, api_key=GROQ_API_KEY, streaming=True)

    if provider == "cerebras":
        # Note: This custom class does not support streaming.
        return CustomChatCerebras(model_name=model_name)
        
    raise ValueError(f"Provider '{provider}' not configured.")