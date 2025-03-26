import asyncio
import os
from dotenv import load_dotenv

from saf_eval.config import Config
from saf_eval.core.pipeline import EvaluationPipeline
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.containment.checker import ContainmentChecker
from saf_eval.llm.providers.openai import OpenAILLM
from saf_eval.retrieval.providers.simple import SimpleRetriever
from saf_eval.evaluation.classifier import FactClassifier
from saf_eval.evaluation.scoring import FactualityScorer

# Load environment variables
load_dotenv()

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
        "Moby Dick": "Moby Dick is a novel by Herman Melville published in 1851.",
        "Herman Melville": "Herman Melville was an American novelist and poet, best known for the novel Moby Dick."
    }
    
    # Initialize pipeline components
    extractor = FactExtractor(config=config, llm=llm)
    containment_checker = ContainmentChecker(config=config, llm=llm)
    retriever = SimpleRetriever(config=config, knowledge_base=knowledge_base)
    classifier = FactClassifier(config=config, llm=llm)
    scorer = FactualityScorer(config=config)
    
    # Create pipeline with containment checker
    pipeline = EvaluationPipeline(
        config=config,
        extractor=extractor,
        retriever=retriever,
        classifier=classifier,
        scorer=scorer,
        containment_checker=containment_checker
    )
    
    # Example response with non-self-contained facts
    response = """
    Melville wrote it in the mid-19th century. 
    It's considered one of the Great American Novels.
    The book tells the story of Captain Ahab's quest for revenge.
    """
    
    context = "Information about the novel Moby Dick"
    
    # Run evaluation with context
    result = await pipeline.run(response, context)
    
    # Print results
    print("=== Evaluation Results ===")
    print(f"Factuality Score: {result.factuality_score:.2f}\n")
    print("Facts extracted and processed:")
    
    for i, fact in enumerate(result.facts):
        print(f"\n[{i+1}] Original Fact: {fact.source_text}")
        print(f"    Self-Contained: {fact.is_self_contained}")
        print(f"    Processed Fact: {fact.text}")
    
    print("\nFact Evaluations:")
    for i, eval in enumerate(result.evaluations):
        print(f"\n[{i+1}] Fact: {eval.fact.text}")
        print(f"    Category: {eval.category}")
        print(f"    Confidence: {eval.confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
