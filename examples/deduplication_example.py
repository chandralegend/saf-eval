import asyncio
import os
from typing import List
from dotenv import load_dotenv

from saf_eval.config import Config
from saf_eval.core.pipeline import EvaluationPipeline
from saf_eval.core.models import AtomicFact
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.containment.checker import ContainmentChecker
from saf_eval.llm.providers.openai import OpenAILLM
from saf_eval.retrieval.providers.simple import SimpleRetriever
from saf_eval.evaluation.classifier import FactClassifier
from saf_eval.evaluation.scoring import FactualityScorer
from saf_eval.utils.deduplication import deduplicate_facts

# Load environment variables
load_dotenv()

# Custom deduplication function that keeps the shortest version of similar facts
def custom_deduplicator(facts: List[AtomicFact]) -> List[AtomicFact]:
    """Custom deduplication that prefers shorter facts when duplicates are found."""
    from saf_eval.utils.deduplication import _calculate_similarity
    
    if not facts:
        return []
    
    # Sort facts by length (shortest first)
    sorted_facts = sorted(facts, key=lambda f: len(f.text))
    unique_facts = [sorted_facts[0]]
    
    # Compare each fact with the ones we've already kept
    for fact in sorted_facts[1:]:
        is_duplicate = False
        for unique_fact in unique_facts:
            similarity = _calculate_similarity(fact.text, unique_fact.text)
            if similarity >= 0.85:  # Same threshold as default
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_facts.append(fact)
    
    return unique_facts

async def main():
    # Initialize configuration
    config = Config(
        scoring_rubric={
            "supported": 1.0,
            "contradicted": 0.0,
            "unverifiable": 0.5
        }
    )
    
    # Initialize LLM
    llm = OpenAILLM(
        model="gpt-4", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create a simple knowledge base
    knowledge_base = {
        "Paris": "Paris is the capital city of France and is known for landmarks like the Eiffel Tower.",
        "Eiffel Tower": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. It is named after engineer Gustave Eiffel.",
        "France": "France is a country in Western Europe with cities such as Paris, Lyon, and Marseille."
    }
    
    # Initialize pipeline components
    extractor = FactExtractor(config=config, llm=llm)
    containment_checker = ContainmentChecker(config=config, llm=llm)
    retriever = SimpleRetriever(config=config, knowledge_base=knowledge_base)
    classifier = FactClassifier(config=config, llm=llm)
    scorer = FactualityScorer(config=config)
    
    # Create pipeline with deduplication
    pipeline = EvaluationPipeline(
        config=config,
        extractor=extractor,
        retriever=retriever,
        classifier=classifier,
        scorer=scorer,
        containment_checker=containment_checker,
        deduplication_fn=custom_deduplicator  # Use our custom deduplication
    )
    
    # Example response with likely duplicate facts after extraction
    response = """
    The Eiffel Tower is located in Paris, France.
    Paris, the capital of France, is home to the Eiffel Tower.
    The famous Eiffel Tower can be found in Paris.
    The tower was named after Gustave Eiffel.
    """
    
    context = "Information about landmarks in France"
    
    # Run evaluation with deduplication
    result = await pipeline.run(response, context)
    
    # Compare with default deduplication
    default_pipeline = EvaluationPipeline(
        config=config,
        extractor=extractor,
        retriever=retriever,
        classifier=classifier,
        scorer=scorer,
        containment_checker=containment_checker
        # Use default deduplication
    )
    
    facts_with_custom = await extractor.extract_facts(response, context)
    facts_with_custom = custom_deduplicator(facts_with_custom)
    
    facts_with_default = await extractor.extract_facts(response, context)
    facts_with_default = deduplicate_facts(facts_with_default)
    
    # Print results
    print("=== Deduplication Comparison ===")
    
    print("\nOriginal facts extracted:")
    original_facts = await extractor.extract_facts(response, context)
    for i, fact in enumerate(original_facts):
        print(f"[{i+1}] {fact.text}")
    
    print("\nFacts after custom deduplication:")
    for i, fact in enumerate(facts_with_custom):
        print(f"[{i+1}] {fact.text}")
    
    print("\nFacts after default deduplication:")
    for i, fact in enumerate(facts_with_default):
        print(f"[{i+1}] {fact.text}")
    
    print("\n=== Final Evaluation Results ===")
    print(f"Factuality Score: {result.factuality_score:.2f}")
    print("\nEvaluated Facts:")
    for i, evaluation in enumerate(result.evaluations):
        print(f"\n[{i+1}] Fact: {evaluation.fact.text}")
        print(f"    Category: {evaluation.category}")
        print(f"    Confidence: {evaluation.confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
