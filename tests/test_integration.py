import os
import pytest
from dotenv import load_dotenv

from saf_eval.config import Config
from saf_eval.core.pipeline import EvaluationPipeline
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.containment.checker import ContainmentChecker
from saf_eval.llm.providers.openai import OpenAILLM
from saf_eval.retrieval.providers.simple import SimpleRetriever
from saf_eval.evaluation.classifier import FactClassifier
from saf_eval.evaluation.scoring import FactualityScorer

# Basic integration test that doesn't require any external dependencies
def test_pipeline_integration():
    # Placeholder for simple pipeline integration test
    assert True

# Skip this test if no API key is available or if running in CI
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None or os.environ.get("CI") is not None,
    reason="Requires OpenAI API key and should not run in CI"
)
async def test_openai_integration():
    """Full pipeline integration test using actual OpenAI API."""
    # Load API key from environment or .env file
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        pytest.skip("OpenAI API key not available")

    # Initialize configuration
    config = Config(
        scoring_rubric={
            "supported": 1.0,
            "contradicted": 0.0,
            "unverifiable": 0.5
        },
        retrieval_config={"top_k": 2},
        llm_config={"temperature": 0.0}  # Use deterministic responses for testing
    )
    
    # Initialize LLM
    llm = OpenAILLM(
        model="gpt-3.5-turbo",  # Use cheaper model for tests
        api_key=api_key
    )
    
    # Create a simple knowledge base with verifiable facts
    knowledge_base = {
        "Earth": "Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
        "Moon": "The Moon is Earth's only natural satellite and orbits Earth at an average distance of 384,400 km."
    }
    
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
    
    # Simple response with factual statements
    response = "Earth is the third planet from the Sun. The Moon orbits around Earth."
    context = "Facts about our solar system"
    
    # Run evaluation
    result = await pipeline.run(response, context)
    
    # Validate the results
    assert result is not None
    assert len(result.facts) > 0
    assert len(result.evaluations) > 0
    assert 0 <= result.factuality_score <= 1
    
    # Validate that facts were properly processed
    for fact in result.facts:
        assert fact.text is not None
        assert fact.is_self_contained is not None
    
    # Validate classifications
    for evaluation in result.evaluations:
        assert evaluation.category in config.evaluation_categories
        assert 0 <= evaluation.confidence <= 1
        assert len(evaluation.documents) <= config.retrieval_config["top_k"]
