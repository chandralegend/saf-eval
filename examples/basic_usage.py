import asyncio
import os
from dotenv import load_dotenv

from saf_eval.config import Config
from saf_eval.core.pipeline import EvaluationPipeline
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.llm.providers.openai import OpenAILLM
from saf_eval.retrieval.providers.simple import SimpleRetriever
from saf_eval.evaluation.classifier import FactClassifier
from saf_eval.evaluation.scoring import FactualityScorer

# Load environment variables
load_dotenv()

async def main():
    # Initialize configuration
    config = Config(
        evaluation_categories=["supported", "contradicted", "unverifiable"],
        scoring_rubric={
            "supported": 1.0,
            "contradicted": 0.0,
            "unverifiable": 0.5
        }
    )
    
    # Initialize components
    llm = OpenAILLM(
        model="gpt-4", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create a simple knowledge base for this example
    knowledge_base = {
        "The Great Wall of China is more than 13,000 miles long.": "The Great Wall of China is actually about 5,500 miles (8,850 km) long.",
        "Mount Everest is the tallest mountain in the world.": "Mount Everest is the highest mountain above sea level at 29,035 feet (8,850 meters)."
    }
    
    extractor = FactExtractor(llm=llm)
    retriever = SimpleRetriever(knowledge_base=knowledge_base)
    classifier = FactClassifier(llm=llm, categories=config.evaluation_categories)
    scorer = FactualityScorer(scoring_rubric=config.scoring_rubric)
    
    # Create the evaluation pipeline
    pipeline = EvaluationPipeline(
        config=config,
        extractor=extractor,
        retriever=retriever,
        classifier=classifier,
        scorer=scorer
    )
    
    # Sample AI response to evaluate
    response = """
    The Great Wall of China stretches for over 13,000 miles across China.
    Mount Everest, standing at 29,035 feet, is the highest mountain above sea level.
    """
    
    context = "Facts about notable landmarks and geographical features"
    
    # Run the evaluation
    result = await pipeline.run(response, context)
    
    # Print the results
    print(f"Factuality Score: {result.factuality_score:.2f}")
    print("\nEvaluated Facts:")
    
    for i, evaluation in enumerate(result.evaluations):
        print(f"\n[{i+1}] Fact: {evaluation.fact.text}")
        print(f"    Category: {evaluation.category}")
        print(f"    Confidence: {evaluation.confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
