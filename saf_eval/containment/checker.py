from typing import List
from ..core.models import AtomicFact
from ..llm.base import LLMBase

class ContainmentChecker:
    """Checks if atomic facts are self-contained within the response."""
    
    def __init__(self, llm: LLMBase = None):
        self.llm = llm
    
    async def check_containment(self, facts: List[AtomicFact], response: str) -> List[AtomicFact]:
        """Check if each fact is self-contained within the response."""
        if self.llm:
            return await self._check_with_llm(facts, response)
        else:
            return self._check_basic(facts, response)
    
    async def _check_with_llm(self, facts: List[AtomicFact], response: str) -> List[AtomicFact]:
        """Use LLM to determine if facts are self-contained."""
        updated_facts = []
        
        for fact in facts:
            prompt = f"""
            Determine if the following fact is self-contained within the original text.
            A self-contained fact is one that doesn't require external context to understand.
            
            Original text: {response}
            
            Fact to check: {fact.text}
            
            Is this fact self-contained? Answer with only 'yes' or 'no'.
            """
            
            result = await self.llm.generate(prompt)
            is_self_contained = result.strip().lower() == 'yes'
            
            # Create a new fact with updated containment status
            updated_fact = fact.model_copy(update={"is_self_contained": is_self_contained})
            updated_facts.append(updated_fact)
        
        return updated_facts
    
    def _check_basic(self, facts: List[AtomicFact], response: str) -> List[AtomicFact]:
        """Basic check assuming all facts from the response are self-contained."""
        updated_facts = []
        
        for fact in facts:
            # A simple check - if all words in the fact are in the response
            fact_words = set(fact.text.lower().split())
            response_words = set(response.lower().split())
            is_self_contained = fact_words.issubset(response_words)
            
            # Create a new fact with updated containment status
            updated_fact = fact.model_copy(update={"is_self_contained": is_self_contained})
            updated_facts.append(updated_fact)
        
        return updated_facts
