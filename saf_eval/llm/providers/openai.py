from typing import Dict, Any, Type, TypeVar, Optional
import json
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..base import LLMBase

T = TypeVar('T', bound=BaseModel)

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
        """Generate structured JSON output using OpenAI API (legacy method)."""
        params = {**self.kwargs, **kwargs}
        system_prompt = f"Output valid JSON according to this schema: {json.dumps(json_schema)}"
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            **params
        )
        return json.loads(response.choices[0].message.content)
    
    async def generate_structured(self, 
                                prompt: str, 
                                schema_model: Type[T], 
                                system_prompt: Optional[str] = None, 
                                **kwargs) -> T:
        """
        Generate structured output directly parsed into a Pydantic model.
        
        Args:
            prompt: The user prompt
            schema_model: A Pydantic model class defining the output structure
            system_prompt: Optional system prompt to guide the generation
            **kwargs: Additional parameters to pass to the OpenAI API
            
        Returns:
            An instance of the provided Pydantic model
        """
        params = {**self.kwargs, **kwargs}
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        completion = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=schema_model,
            **params
        )
        
        return completion.choices[0].message.parsed
