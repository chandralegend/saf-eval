from typing import List

from ..config import Config
from ..extraction.extractor import FactExtractor
from ..retrieval.base import RetrieverBase
from ..evaluation.classifier import FactClassifier
from ..evaluation.scoring import FactualityScorer
from .models import ResponseEvaluation

class EvaluationPipeline:
    """Main pipeline for evaluating factuality of AI responses."""
    
    def __init__(self, config: Config, extractor: FactExtractor, 
                 retriever: RetrieverBase, classifier: FactClassifier,
                 scorer: FactualityScorer):
        self.config = config
        self.extractor = extractor
        self.retriever = retriever
        self.classifier = classifier
        self.scorer = scorer
    
    async def run(self, response: str, context: str) -> ResponseEvaluation:
        """Run the full evaluation pipeline on a response."""
        # Step 1: Extract atomic facts
        facts = await self.extractor.extract_facts(response)
        
        # Step 2: For each fact, retrieve relevant documents
        evaluations = []
        for fact in facts:
            # Step 3: Retrieve documents for the fact
            documents = await self.retriever.retrieve(fact)
            
            # Step 4: Classify the fact based on retrieved documents
            evaluation = await self.classifier.classify(fact, documents)
            evaluations.append(evaluation)
        
        # Step 5: Calculate overall factuality score
        result = self.scorer.score(response, context, evaluations)
        
        return result
