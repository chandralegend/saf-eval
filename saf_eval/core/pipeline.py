from typing import List, Optional

from ..config import Config
from ..extraction.extractor import FactExtractor
from ..containment.checker import ContainmentChecker
from ..retrieval.base import RetrieverBase
from ..evaluation.classifier import FactClassifier
from ..evaluation.scoring import FactualityScorer
from .models import ResponseEvaluation

class EvaluationPipeline:
    """Main pipeline for evaluating factuality of AI responses."""
    
    def __init__(self, config: Config, extractor: FactExtractor, 
                 retriever: RetrieverBase, classifier: FactClassifier,
                 scorer: FactualityScorer, containment_checker: Optional[ContainmentChecker] = None):
        self.config = config
        self.extractor = extractor
        self.retriever = retriever
        self.classifier = classifier
        self.scorer = scorer
        self.containment_checker = containment_checker
    
    async def run(self, response: str, context: str = None) -> ResponseEvaluation:
        """Run the full evaluation pipeline on a response."""
        # Step 1: Extract atomic facts
        facts = await self.extractor.extract_facts(response, context)
        
        # Step 2: Check if facts are self-contained and make them self-contained if needed
        if self.containment_checker:
            # First check which facts are self-contained
            facts = await self.containment_checker.check_containment(facts, response)
            # Then make non-self-contained facts self-contained
            facts = await self.containment_checker.self_contain_facts(facts, response, context)
        
        # Step 3: For each fact, retrieve relevant documents
        evaluations = []
        for fact in facts:
            # Step 4: Retrieve documents for the fact
            documents = await self.retriever.retrieve(fact)
            
            # Step 5: Classify the fact based on retrieved documents
            evaluation = await self.classifier.classify(fact, documents)
            evaluations.append(evaluation)
        
        # Step 6: Calculate overall factuality score
        result = self.scorer.score(response, context, evaluations)
        
        return result
