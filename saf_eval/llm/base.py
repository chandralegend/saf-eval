from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class LLMBase(ABC):
    """Base class for LLM implementations."""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the prompt."""
        pass
    
    @abstractmethod
    async def generate_with_json_output(self, prompt: str, json_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured output as JSON."""
        pass
    
    @abstractmethod
    async def generate_structured(self, prompt: str, schema_model: Type[T], 
                               system_prompt: Optional[str] = None, **kwargs) -> T:
        """
        Generate structured output parsed into a Pydantic model.
        
        Args:
            prompt: The user prompt
            schema_model: A Pydantic model class defining the output structure
            system_prompt: Optional system prompt to guide the generation
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            An instance of the provided Pydantic model
        """
        pass
