from typing import Dict, List

from ..config import Config
from ..core.models import AtomicFact, RetrievedDocument, FactEvaluation
from ..llm.base import LLMBase

class FactClassifier:
    """Classifies facts into categories based on retrieved documents."""
    
    def __init__(self, config: Config, llm: LLMBase):
        self.config = config
        self.llm = llm
    
    async def classify(self, fact: AtomicFact, documents: List[RetrievedDocument]) -> FactEvaluation:
        """Classify a fact based on the retrieved documents."""
        prompt = self._build_classification_prompt(fact, documents)
        
        classification_schema = {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": self.config.evaluation_categories},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
        
        result = await self.llm.generate_with_json_output(prompt, classification_schema)
        
        return FactEvaluation(
            fact=fact,
            documents=documents,
            category=result["category"],
            confidence=result["confidence"]
        )
    
    def _build_classification_prompt(self, fact: AtomicFact, documents: List[RetrievedDocument]) -> str:
        """Build the prompt for fact classification."""
        doc_texts = "\n\n".join([f"Document {i+1}: {doc.content}" for i, doc in enumerate(documents)])
        
        return f"""
        Classify the following fact based on the retrieved documents.
        
        Fact: {fact.text}
        
        Retrieved Documents:
        {doc_texts}
        
        Available categories: {', '.join(self.config.evaluation_categories)}
        
        Provide your classification as JSON with 'category' and 'confidence' (0-1) fields.
        """
