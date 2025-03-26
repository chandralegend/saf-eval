import asyncio
import os
from typing import List
import uuid

from dotenv import load_dotenv

from saf_eval.config import Config
from saf_eval.core.pipeline import EvaluationPipeline
from saf_eval.core.models import AtomicFact, RetrievedDocument
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.llm.providers.openai import OpenAILLM
from saf_eval.retrieval.base import RetrieverBase
from saf_eval.evaluation.classifier import FactClassifier
from saf_eval.evaluation.scoring import FactualityScorer

# Load environment variables
load_dotenv()

# Custom retriever implementation using a vector database
class VectorDBRetriever(RetrieverBase):
    """Custom retriever that simulates a vector database search."""
    
    def __init__(self, config: Config, vector_db_client=None):
        super().__init__(config)
        self.vector_db_client = vector_db_client or self._mock_vector_db()
        self.top_k = self.config.retrieval_config.get("top_k", 3)
        
    async def retrieve(self, fact: AtomicFact, **kwargs) -> List[RetrievedDocument]:
        """Retrieve documents from vector database."""
        # In a real implementation, this would embed the fact and query the vector DB
        query = fact.text
        results = self.vector_db_client.search(query, self.top_k)
        
        documents = []
        for i, result in enumerate(results):
            documents.append(
                RetrievedDocument(
                    id=f"doc-{uuid.uuid4()}",
                    content=result["content"],
                    source=result["source"],
                    relevance_score=result["score"]
                )
            )
        
        return documents
    
    def _mock_vector_db(self):
        """Create a mock vector database client for demonstration."""
        class MockVectorDB:
            def __init__(self):
                self.documents = [
                    {
                        "content": "The Great Wall of China is approximately 5,500 miles (8,850 km) long.",
                        "source": "Encyclopedia of World Geography",
                        "embedding": [0.1, 0.2, 0.3]  # Simplified embedding
                    },
                    {
                        "content": "Mount Everest reaches a height of 29,032 feet (8,849 meters).",
                        "source": "National Geographic",
                        "embedding": [0.4, 0.5, 0.6]
                    },
                    {
                        "content": "The Great Wall of China was built over many centuries, with the most famous sections dating to the Ming Dynasty.",
                        "source": "History Journal",
                        "embedding": [0.7, 0.8, 0.9]
                    }
                ]
            
            def search(self, query, top_k):
                # In a real implementation, this would compute cosine similarity
                # For this example, we'll use simple keyword matching
                results = []
                for doc in self.documents:
                    # Naive keyword matching for demo purposes
                    score = 0.0
                    if any(keyword in doc["content"].lower() for keyword in query.lower().split()):
                        score = 0.8  # Arbitrary score for matching documents
                    
                    if score > 0:
                        results.append({
                            "content": doc["content"],
                            "source": doc["source"],
                            "score": score
                        })
                
                # Sort by score and limit to top_k
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:top_k]
        
        return MockVectorDB()

async def main():
    # Initialize configuration with retrieval settings
    config = Config(
        scoring_rubric={
            "supported": 1.0,
            "contradicted": 0.0,
            "unverifiable": 0.5
        },
        retrieval_config={
            "top_k": 2,
            "embedding_model": "text-embedding-3-small"  # Example of custom retrieval config
        }
    )
    
    # Initialize components with shared config
    llm = OpenAILLM(
        model="gpt-4", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Setup pipeline components
    extractor = FactExtractor(config=config, llm=llm)
    # Use our custom vector DB retriever
    retriever = VectorDBRetriever(config=config)
    classifier = FactClassifier(config=config, llm=llm)
    scorer = FactualityScorer(config=config)
    
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
    The Great Wall of China is more than 13,000 miles long.
    Mount Everest stands at approximately 29,035 feet above sea level.
    """
    
    context = "Facts about world landmarks"
    
    # Run the evaluation
    result = await pipeline.run(response, context)
    
    # Print the results
    print(f"Factuality Score: {result.factuality_score:.2f}")
    print("\nEvaluated Facts:")
    
    for i, evaluation in enumerate(result.evaluations):
        print(f"\n[{i+1}] Fact: {evaluation.fact.text}")
        print(f"    Category: {evaluation.category}")
        print(f"    Confidence: {evaluation.confidence:.2f}")
        print(f"    Retrieved documents:")
        for j, doc in enumerate(evaluation.documents):
            print(f"      - Document {j+1}: {doc.content} (Source: {doc.source}, Score: {doc.relevance_score:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
