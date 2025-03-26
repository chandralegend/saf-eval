from typing import List, Optional
from ..config import Config
from ..core.models import AtomicFact
from ..llm.base import LLMBase

class ContainmentChecker:
    """Checks if atomic facts are self-contained within the response and fixes those that aren't."""
    
    def __init__(self, config: Config, llm: LLMBase = None):
        self.config = config
        self.llm = llm
    
    async def check_containment(self, facts: List[AtomicFact], response: str) -> List[AtomicFact]:
        """Check if each fact is self-contained within the response."""
        if self.llm:
            return await self._check_with_llm(facts, response)
        else:
            return self._check_basic(facts, response)
    
    async def self_contain_facts(self, facts: List[AtomicFact], response: str, context: Optional[str] = None) -> List[AtomicFact]:
        """
        Make non-self-contained facts self-contained by adding necessary context.
        
        Args:
            facts: List of atomic facts to process
            response: The original response text
            context: Optional additional context
            
        Returns:
            List of self-contained atomic facts
        """
        if not self.llm:
            # Without LLM, we can't reliably self-contain facts
            return facts
            
        updated_facts = []
        
        for fact in facts:
            # Skip already self-contained facts
            if fact.is_self_contained:
                updated_facts.append(fact)
                continue
                
            prompt = f"""
            Make the following fact self-contained so it can be understood without additional context.
            A self-contained fact should include all necessary details to be understood on its own.
            
            Original response: {response}
            
            """
            
            if context:
                prompt += f"Additional context: {context}\n\n"
                
            prompt += f"""
            Fact to make self-contained: {fact.text}
            
            Rewrite this as a self-contained fact:
            """
            
            result = await self.llm.generate(prompt)
            self_contained_text = result.strip()
            
            # Create a new fact with the self-contained text
            updated_fact = fact.model_copy(
                update={
                    "text": self_contained_text,
                    "is_self_contained": True
                }
            )
            updated_facts.append(updated_fact)
            
        return updated_facts
    
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
