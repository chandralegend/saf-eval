from typing import List
from ..core.models import AtomicFact
from ..llm.base import LLMBase

class RelevancyChecker:
    """Checks if atomic facts are relevant to the given context."""
    
    def __init__(self, llm: LLMBase = None):
        self.llm = llm
    
    async def check_relevancy(self, facts: List[AtomicFact], context: str) -> List[AtomicFact]:
        """Check if each fact is relevant to the context."""
        if self.llm:
            return await self._check_with_llm(facts, context)
        else:
            return self._check_basic(facts, context)
    
    async def _check_with_llm(self, facts: List[AtomicFact], context: str) -> List[AtomicFact]:
        """Use LLM to determine if facts are relevant to the context."""
        updated_facts = []
        
        for fact in facts:
            prompt = f"""
            Determine if the following fact is relevant to the given context.
            
            Context: {context}
            
            Fact: {fact.text}
            
            Is this fact relevant to the context? Answer with only 'yes' or 'no'.
            """
            
            result = await self.llm.generate(prompt)
            is_relevant = result.strip().lower() == 'yes'
            
            # Create a new fact with updated relevancy status
            updated_fact = fact.model_copy(update={"is_relevant": is_relevant})
            updated_facts.append(updated_fact)
        
        return updated_facts
    
    def _check_basic(self, facts: List[AtomicFact], context: str) -> List[AtomicFact]:
        """Basic check using keyword overlap between facts and context."""
        updated_facts = []
        context_words = set(context.lower().split())
        
        for fact in facts:
            fact_words = set(fact.text.lower().split())
            # Calculate word overlap as a simple measure of relevance
            overlap = fact_words.intersection(context_words)
            is_relevant = len(overlap) > 0
            
            # Create a new fact with updated relevancy status
            updated_fact = fact.model_copy(update={"is_relevant": is_relevant})
            updated_facts.append(updated_fact)
        
        return updated_facts
