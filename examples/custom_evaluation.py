import asyncio
import os
from typing import List
from dotenv import load_dotenv

from saf_eval.config import Config
from saf_eval.core.pipeline import EvaluationPipeline
from saf_eval.core.models import AtomicFact, RetrievedDocument, FactEvaluation, ResponseEvaluation
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.llm.providers.openai import OpenAILLM
from saf_eval.retrieval.providers.simple import SimpleRetriever
from saf_eval.evaluation.classifier import FactClassifier
from saf_eval.evaluation.scoring import FactualityScorer

# Load environment variables
load_dotenv()

# Custom classifier with enhanced categories
class EnhancedFactClassifier(FactClassifier):
    """Enhanced fact classifier with more detailed categories."""
    
    def __init__(self, config: Config, llm: OpenAILLM):
        super().__init__(config=config, llm=llm)
    
    def _build_classification_prompt(self, fact: AtomicFact, documents: List[RetrievedDocument]) -> str:
        """Build an enhanced prompt with detailed category descriptions."""
        doc_texts = "\n\n".join([f"Document {i+1}: {doc.content}" for i, doc in enumerate(documents)])
        
        categories_description = """
        Categories:
        - fully_supported: The fact is completely supported by the documents with high confidence
        - partially_supported: Some aspects of the fact are supported, but not all details
        - no_evidence: The documents do not provide enough information to verify the fact
        - contradicted: The documents explicitly contradict the fact
        - misleading: The fact is technically correct but presents information in a misleading way
        """
        
        return f"""
        Classify the following fact based on the retrieved documents.
        
        Fact: {fact.text}
        
        Retrieved Documents:
        {doc_texts}
        
        {categories_description}
        
        Available categories: {', '.join(self.config.evaluation_categories)}
        
        Provide your classification as JSON with 'category' and 'confidence' (0-1) fields.
        """

# Custom scoring system with weighted categories
class EnhancedFactualityScorer(FactualityScorer):
    """Enhanced factuality scorer with weighted categories."""
    
    def __init__(self, config: Config):
        super().__init__(config=config)
    
    def score(self, response_text: str, context: str, evaluations: List[FactEvaluation]) -> ResponseEvaluation:
        """Calculate weighted factuality score with baseline adjustment."""
        result = super().score(response_text, context, evaluations)
        
        # For negative scores (due to contradictions/misleading), normalize to 0-1 range
        if result.factuality_score < 0:
            result.factuality_score = 0
        
        return result

async def main():
    # Initialize configuration with enhanced scoring rubric
    config = Config(
        scoring_rubric={
            "fully_supported": 1.0,
            "partially_supported": 0.6,
            "no_evidence": 0.3,
            "contradicted": -1.0,  # Penalize contradictions
            "misleading": -0.5     # Penalize misleading facts
        }
    )
    
    # Initialize components
    llm = OpenAILLM(
        model="gpt-4", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create a knowledge base for this example
    knowledge_base = {
        "The Eiffel Tower": "The Eiffel Tower is 330 meters (1,083 ft) tall and was completed in 1889.",
        "The Mona Lisa": "The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519.",
        "The Amazon River": "The Amazon River is approximately 6,400 km (4,000 miles) long and is the largest river by discharge volume.",
    }
    
    # Set up pipeline components with shared config
    extractor = FactExtractor(config=config, llm=llm)
    retriever = SimpleRetriever(config=config, knowledge_base=knowledge_base)
    
    # Use our enhanced classifier and scorer
    classifier = EnhancedFactClassifier(config=config, llm=llm)
    scorer = EnhancedFactualityScorer(config=config)
    
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
    The Eiffel Tower stands at exactly 1,000 feet tall and was built in the late 19th century.
    The Mona Lisa was painted by Leonardo da Vinci in the early 16th century.
    The Amazon River spans more than 6,500 kilometers, making it the longest river in the world.
    """
    
    context = "Facts about famous landmarks and natural features"
    
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
