import uuid
from typing import Dict, List
import re

from ...config import Config
from ...core.models import AtomicFact, RetrievedDocument
from ..base import RetrieverBase

class SimpleRetriever(RetrieverBase):
    """Simple retriever using keyword matching against a knowledge base."""
    
    def __init__(self, config: Config, knowledge_base: Dict[str, str]):
        super().__init__(config)
        self.knowledge_base = knowledge_base
        self.top_k = self.config.retrieval_config.get("top_k", 3)
    
    async def retrieve(self, fact: AtomicFact, **kwargs) -> List[RetrievedDocument]:
        """Retrieve documents from the knowledge base based on keyword matching."""
        query_terms = self._extract_keywords(fact.text)
        matching_documents = []
        
        for key, content in self.knowledge_base.items():
            # Simple keyword matching
            relevance = self._calculate_relevance(query_terms, key + " " + content)
            if relevance > 0:
                doc = RetrievedDocument(
                    id=str(uuid.uuid4()),
                    content=content,
                    source=key,
                    relevance_score=relevance
                )
                matching_documents.append(doc)
        
        # Sort by relevance score and return
        matching_documents.sort(key=lambda x: x.relevance_score, reverse=True)
        return matching_documents[:self.top_k]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Remove common stop words and punctuation
        stop_words = {"a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "by", "and", "or", "for", "with", "to", "from"}
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _calculate_relevance(self, query_terms: List[str], document: str) -> float:
        """Calculate relevance score based on term frequency."""
        document = document.lower()
        term_count = sum(1 for term in query_terms if term in document)
        
        if not term_count:
            return 0
        
        # Simple relevance calculation based on percentage of matching terms
        relevance = term_count / len(query_terms) if query_terms else 0
        return relevance
