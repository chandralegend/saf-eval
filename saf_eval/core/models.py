from typing import Dict, List, Optional
from pydantic import BaseModel

class AtomicFact(BaseModel):
    """A single, atomic fact extracted from a response."""
    id: str
    text: str
    source_text: str
    is_self_contained: Optional[bool] = None
    is_relevant: Optional[bool] = None
    
class RetrievedDocument(BaseModel):
    """A document retrieved for fact verification."""
    id: str
    content: str
    source: str
    relevance_score: Optional[float] = None
    
class FactEvaluation(BaseModel):
    """Evaluation result for an atomic fact."""
    fact: AtomicFact
    documents: List[RetrievedDocument]
    category: str
    confidence: float
    
class ResponseEvaluation(BaseModel):
    """Overall evaluation for an AI response."""
    response_text: str
    context: str
    facts: List[AtomicFact]
    evaluations: List[FactEvaluation]
    factuality_score: float
