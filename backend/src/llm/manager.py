"""
MODELS=[
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite-preview",
    "gemma-4-31b-it"
]
"""

import json
import re
from src.llm.client import GeminiClient


class LLMManager:
    def __init__(self):
        self._MODEL = [
            {
                "name": "gemini-3.1-pro-preview",
                "client": GeminiClient("gemini-3.1-pro-preview"),
                "is_available": True,
            },
            {
                "name": "gemini-2.5-flash",
                "client": GeminiClient("gemini-2.5-flash"),
                "is_available": True,
            },
            {
                "name": "gemini-3.1-flash-lite-preview",
                "client": GeminiClient("gemini-3.1-flash-lite-preview"),
                "is_available": True,
            },
            {
                "name": "gemma-4-31b-it",
                "client": GeminiClient("gemma-4-31b-it"),
                "is_available": True,
            },
        ]

    def _safe_parse(self, text: str) -> dict:
        """
        Robustly parse JSON from Gemini's response.
        Handles unescaped control characters (newlines inside string values)
        that occasionally slip through even with response_mime_type=application/json.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Replace literal unescaped control characters inside the raw string
            # (tab, newline, carriage return) with their escaped equivalents,
            # but only when they appear inside JSON string values.
            print(f"Throw Error {e}")
            cleaned = re.sub(
                r"(?<!\\)([\x00-\x1f\x7f])", 
                lambda m: repr(m.group()), 
                text
            )
            return json.load(cleaned)

    async def manage_model(self, prompt: str):
        for model in self._MODEL:
            if not model["is_available"]:
                continue
            try:
                raw_json = await model["client"].generate_feedback(prompt)
                feedback = self._safe_parse(raw_json)
                return feedback
            except Exception as e:
                print(f"⚠️ Model {model['name']} failed: {e}.")
                model["is_available"] = False
                continue

        raise Exception("All models are down")
