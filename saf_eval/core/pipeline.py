from typing import List, Optional, Callable
import time
import uuid

from ..config import Config
from ..extraction.extractor import FactExtractor
from ..containment.checker import ContainmentChecker
from ..retrieval.base import RetrieverBase
from ..evaluation.classifier import FactClassifier
from ..evaluation.scoring import FactualityScorer
from ..utils.deduplication import deduplicate_facts
from ..utils.logging import get_logger
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
        
        # Initialize logger
        self.logger = get_logger(
            name="pipeline",
            level=self.config.logging.level,
            log_dir=self.config.logging.log_dir,
            console=self.config.logging.console,
            file=self.config.logging.file,
            json_format=self.config.logging.json_format
        )
    
    async def run(self, response: str, context: str = None) -> ResponseEvaluation:
        """Run the full evaluation pipeline on a response."""
        evaluation_id = str(uuid.uuid4())
        start_time = time.time()
        self.logger.info(f"Starting evaluation pipeline", {
            "evaluation_id": evaluation_id,
            "response_length": len(response),
            "has_context": context is not None
        })
        
        # Step 1: Extract atomic facts
        self.logger.info(f"Extracting atomic facts", {"evaluation_id": evaluation_id})
        facts = await self.extractor.extract_facts(response, context)
        self.logger.info(f"Extracted {len(facts)} atomic facts", {
            "evaluation_id": evaluation_id,
            "fact_count": len(facts)
        })
        
        # Step 2: Check if facts are self-contained and make them self-contained if needed
        if self.containment_checker:
            self.logger.info(f"Checking fact self-containment", {"evaluation_id": evaluation_id})
            # First check which facts are self-contained
            facts = await self.containment_checker.check_containment(facts, response)
            
            # Log self-containment results
            contained_count = sum(1 for f in facts if f.is_self_contained)
            non_contained_count = len(facts) - contained_count
            self.logger.info(f"Self-containment check completed", {
                "evaluation_id": evaluation_id,
                "self_contained_count": contained_count,
                "non_self_contained_count": non_contained_count
            })
            
            # Then make non-self-contained facts self-contained
            if non_contained_count > 0:
                self.logger.info(f"Fixing {non_contained_count} non-self-contained facts", {
                    "evaluation_id": evaluation_id
                })
                facts = await self.containment_checker.self_contain_facts(facts, response, context)
                self.logger.info(f"Fixed non-self-contained facts", {"evaluation_id": evaluation_id})
        
        # Step 3: Deduplicate facts
        original_fact_count = len(facts)
        self.logger.info(f"Deduplicating {original_fact_count} facts", {"evaluation_id": evaluation_id})
        facts = self.deduplication_fn(facts)
        self.logger.info(f"Deduplication complete: {len(facts)}/{original_fact_count} facts remain", {
            "evaluation_id": evaluation_id,
            "facts_before": original_fact_count,
            "facts_after": len(facts),
            "duplicates_removed": original_fact_count - len(facts)
        })
        
        # Step 4: For each fact, retrieve relevant documents and classify
        self.logger.info(f"Starting document retrieval and classification for {len(facts)} facts", 
                        {"evaluation_id": evaluation_id})
        evaluations = []
        for i, fact in enumerate(facts):
            # Log the current fact being processed
            self.logger.debug(f"Processing fact {i+1}/{len(facts)}", {
                "evaluation_id": evaluation_id,
                "fact_index": i,
                "fact_text": fact.text
            })
            
            # Step 5: Retrieve documents for the fact
            documents = await self.retriever.retrieve(fact)
            self.logger.debug(f"Retrieved {len(documents)} documents for fact {i+1}", {
                "evaluation_id": evaluation_id,
                "fact_index": i,
                "document_count": len(documents)
            })
            
            # Step 6: Classify the fact based on retrieved documents
            evaluation = await self.classifier.classify(fact, documents)
            self.logger.debug(f"Classified fact {i+1} as '{evaluation.category}' with confidence {evaluation.confidence:.2f}", {
                "evaluation_id": evaluation_id,
                "fact_index": i,
                "category": evaluation.category,
                "confidence": evaluation.confidence
            })
            
            evaluations.append(evaluation)
        
        # Step 7: Calculate overall factuality score
        self.logger.info(f"Calculating factuality score for {len(evaluations)} evaluations", 
                        {"evaluation_id": evaluation_id})
        result = self.scorer.score(response, context, evaluations)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Evaluation pipeline completed in {elapsed_time:.2f}s with factuality score {result.factuality_score:.2f}", {
            "evaluation_id": evaluation_id,
            "elapsed_time": elapsed_time,
            "factuality_score": result.factuality_score
        })
        
        return result
