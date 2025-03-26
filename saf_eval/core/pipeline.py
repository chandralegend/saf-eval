from typing import List, Optional, Callable

from ..config import Config
from ..extraction.extractor import FactExtractor
from ..containment.checker import ContainmentChecker
from ..retrieval.base import RetrieverBase
from ..evaluation.classifier import FactClassifier
from ..evaluation.scoring import FactualityScorer
from ..utils.deduplication import deduplicate_facts
from .models import ResponseEvaluation, AtomicFact

class EvaluationPipeline:
    """Main pipeline for evaluating factuality of AI responses."""
    
    def __init__(self, config: Config, extractor: FactExtractor, 
                 retriever: RetrieverBase, classifier: FactClassifier,
                 scorer: FactualityScorer, containment_checker: Optional[ContainmentChecker] = None,
                 deduplication_fn: Optional[Callable[[List[AtomicFact]], List[AtomicFact]]] = None):
        self.config = config
        self.extractor = extractor
        self.retriever = retriever
        self.classifier = classifier
        self.scorer = scorer
        self.containment_checker = containment_checker
        self.deduplication_fn = deduplication_fn or deduplicate_facts
    
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
        
        # Step 3: Deduplicate facts (new step)
        facts = self.deduplication_fn(facts)
        
        # Step 4: For each fact, retrieve relevant documents
        evaluations = []
        for fact in facts:
            # Step 5: Retrieve documents for the fact
            documents = await self.retriever.retrieve(fact)
            
            # Step 6: Classify the fact based on retrieved documents
            evaluation = await self.classifier.classify(fact, documents)
            evaluations.append(evaluation)
        
        # Step 7: Calculate overall factuality score
        result = self.scorer.score(response, context, evaluations)
        
        return result
