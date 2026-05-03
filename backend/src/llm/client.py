from abc import ABC, abstractmethod
from google import genai
from google.genai import types

"""
MODELS=[
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite-preview",
    "gemma-4-31b-it"
]
"""
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class CoachFeedback(BaseModel):
    problem: str = Field(
        description="Describe the core biomechanical issue in a short, clear paragraph. DO NOT use markdown, bullet points, or bold text."
    )
    improvement: list[str] = Field(
        description="Provide a step-by-step action plan to fix the issue. Return as a list of independent, concise steps. DO NOT use numbered lists, bullet points, or markdown formatting."
    )
    power_technique: str = Field(
        description="Specific tips on generating explosive power. Write as a direct statement. DO NOT wrap the text in quotation marks. DO NOT use markdown."
    )


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate_feedback(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class GeminiClient(BaseLLMClient):
    def __init__(self, model_name: str, api_key: str = None):
        api_key = api_key or os.environ["GEMINI_API_KEY"]
        if not api_key:
            raise ValueError(f"API Key for {model_name} is missing!")
        self._model_name = model_name
        self.client = genai.Client(api_key=api_key)

    async def generate_feedback(self, prompt: str) -> str:

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                response_schema=CoachFeedback,
                temperature = 0.3
            ),
        )
        return response.text

    @property
    def model_name(self):
        return self._model_name


if __name__ == "__main__":
    from google import genai

    client = genai.Client()
    for model in client.models.list():
        print(model.name)
