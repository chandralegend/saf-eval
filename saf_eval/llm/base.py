from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

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
