import asyncio
import os
from dotenv import load_dotenv

from saf_eval.config import Config, LoggingConfig
from saf_eval.core.pipeline import EvaluationPipeline
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.containment.checker import ContainmentChecker
from saf_eval.llm.providers.openai import OpenAILLM
from saf_eval.retrieval.providers.simple import SimpleRetriever
from saf_eval.evaluation.classifier import FactClassifier
from saf_eval.evaluation.scoring import FactualityScorer
from saf_eval.utils.logging import get_logger

# Load environment variables
load_dotenv()

async def main():
    # Configure logging
    logging_config = LoggingConfig(
        level="DEBUG",
        console=True,
        file=True,
        log_dir="./logs",
        json_format=True
    )
    
    # Initialize configuration with logging settings
    config = Config(
        scoring_rubric={
            "supported": 1.0,
            "contradicted": 0.0,
            "unverifiable": 0.5
        },
        logging=logging_config
    )
    
    # Create a logger for this script
    logger = get_logger(
        name="example-script",
        level=config.logging.level,
        log_dir=config.logging.log_dir,
        console=config.logging.console,
        file=config.logging.file,
        json_format=config.logging.json_format
    )
    
    logger.info("Starting factuality evaluation example with detailed logging")
    
    # Initialize LLM
    llm = OpenAILLM(
        model="gpt-4", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create a knowledge base
    knowledge_base = {
        "Eiffel Tower": "The Eiffel Tower is 330 meters tall and was completed in 1889.",
        "Golden Gate Bridge": "The Golden Gate Bridge was completed in 1937 and has a main span of 1,280 meters.",
        "Statue of Liberty": "The Statue of Liberty was dedicated in 1886 and stands 93 meters tall including the pedestal."
    }
    
    logger.info("Initializing pipeline components")
    
    # Initialize pipeline components
    extractor = FactExtractor(config=config, llm=llm)
    containment_checker = ContainmentChecker(config=config, llm=llm)
    retriever = SimpleRetriever(config=config, knowledge_base=knowledge_base)
    classifier = FactClassifier(config=config, llm=llm)
    scorer = FactualityScorer(config=config)
    
    # Create pipeline
    pipeline = EvaluationPipeline(
        config=config,
        extractor=extractor,
        retriever=retriever,
        classifier=classifier,
        scorer=scorer,
        containment_checker=containment_checker
    )
    
    # Example response with facts that require self-containment
    response = """
    It was built in the late 19th century and stands over 300 meters tall.
    The bridge spans across the Golden Gate Strait connecting San Francisco to Marin County.
    The famous statue was a gift from France to the United States.
    """
    
    context = "Information about famous landmarks"
    
    logger.info("Running evaluation pipeline on response")
    
    # Run evaluation with context
    result = await pipeline.run(response, context)
    
    # Print results
    logger.info(f"Evaluation complete with factuality score: {result.factuality_score:.2f}")
    
    print("\n=== Evaluation Results ===")
    print(f"Factuality Score: {result.factuality_score:.2f}\n")
    print("Processed Facts:")
    
    for i, evaluation in enumerate(result.evaluations):
        print(f"\n[{i+1}] Fact: {evaluation.fact.text}")
        print(f"    Category: {evaluation.category}")
        print(f"    Confidence: {evaluation.confidence:.2f}")
    
    logger.info("Example script completed")

if __name__ == "__main__":
    asyncio.run(main())
