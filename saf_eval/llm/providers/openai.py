from typing import Dict, Any
import json
from openai import AsyncOpenAI

from ..base import LLMBase

class OpenAILLM(LLMBase):
    """OpenAI LLM implementation."""
    
    def __init__(self, model: str = "gpt-4", api_key: str = None, **kwargs):
        super().__init__(model, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        params = {**self.kwargs, **kwargs}
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        return response.choices[0].message.content
    
    async def generate_with_json_output(self, prompt: str, json_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured JSON output using OpenAI API."""
        params = {**self.kwargs, **kwargs}
        system_prompt = f"Output valid JSON according to this schema: {json.dumps(json_schema)}"
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            **params
        )
        return json.loads(response.choices[0].message.content)
