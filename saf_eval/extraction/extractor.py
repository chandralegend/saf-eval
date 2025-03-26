from typing import List
import uuid

from ..config import Config
from ..core.models import AtomicFact
from ..llm.base import LLMBase

class FactExtractor:
    """Extract atomic facts from an AI response."""
    
    def __init__(self, config: Config, llm: LLMBase = None):
        self.config = config
        self.llm = llm
        
    async def extract_facts(self, response: str) -> List[AtomicFact]:
        """Extract atomic facts from the response."""
        if self.llm:
            return await self._extract_with_llm(response)
        else:
            return self._extract_basic(response)
            
    async def _extract_with_llm(self, response: str) -> List[AtomicFact]:
        """Use LLM to extract atomic facts."""
        prompt = f"""
        Extract the atomic facts from the following text. An atomic fact is a simple, 
        self-contained statement that makes a single factual claim.
        
        Text: {response}
        
        Output each atomic fact on a new line.
        """
        
        result = await self.llm.generate(prompt)
        facts = result.strip().split('\n')
        
        return [
            AtomicFact(
                id=str(uuid.uuid4()),
                text=fact.strip(),
                source_text=response
            )
            for fact in facts if fact.strip()
        ]
    
    def _extract_basic(self, response: str) -> List[AtomicFact]:
        """Basic extraction by splitting on periods."""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        return [
            AtomicFact(
                id=str(uuid.uuid4()),
                text=sentence,
                source_text=response
            )
            for sentence in sentences
        ]
