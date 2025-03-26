import pytest
from uuid import uuid4

from saf_eval.core.models import AtomicFact, RetrievedDocument
from saf_eval.retrieval.providers.simple import SimpleRetriever

@pytest.fixture
def knowledge_base():
    return {
        "History": "World War II ended in 1945 with the surrender of Japan.",
        "Science": "The theory of relativity was proposed by Albert Einstein.",
        "Geography": "Mount Everest is the highest mountain above sea level."
    }

@pytest.fixture
def retriever(knowledge_base):
    return SimpleRetriever(knowledge_base=knowledge_base)

@pytest.fixture
def atomic_fact():
    return AtomicFact(
        id=str(uuid4()),
        text="Albert Einstein proposed the theory of relativity in physics.",
        source_text="Albert Einstein proposed the theory of relativity in physics."
    )

@pytest.mark.asyncio
async def test_retrieve(retriever, atomic_fact):
    documents = await retriever.retrieve(atomic_fact)
    
    assert len(documents) > 0
    assert all(isinstance(doc, RetrievedDocument) for doc in documents)
    
    # The Science document should be retrieved with highest relevance
    assert any(doc.content == "The theory of relativity was proposed by Albert Einstein." for doc in documents)
    
    # Check sorting order (most relevant first)
    science_doc = next(doc for doc in documents if "Einstein" in doc.content)
    assert science_doc.relevance_score > 0

@pytest.mark.asyncio
async def test_retrieve_no_match(retriever):
    fact = AtomicFact(
        id=str(uuid4()),
        text="Quantum computing uses qubits instead of classical bits.",
        source_text="Quantum computing uses qubits instead of classical bits."
    )
    
    documents = await retriever.retrieve(fact)
    assert len(documents) == 0  # No matches expected

def test_extract_keywords(retriever):
    text = "The theory of relativity was proposed by Albert Einstein in 1905."
    keywords = retriever._extract_keywords(text)
    
    assert "theory" in keywords
    assert "relativity" in keywords
    assert "albert" in keywords
    assert "einstein" in keywords
    assert "1905" in keywords
    
    # Stop words should be excluded
    assert "the" not in keywords
    assert "of" not in keywords
    assert "was" not in keywords
    assert "by" not in keywords
    assert "in" not in keywords

def test_calculate_relevance(retriever):
    query_terms = ["einstein", "relativity", "physics"]
    document = "The theory of relativity was proposed by Albert Einstein."
    
    relevance = retriever._calculate_relevance(query_terms, document)
    assert relevance > 0
    
    # 2 out of 3 terms match
    assert relevance == pytest.approx(2/3)
    
    # No match
    irrelevant_doc = "The Eiffel Tower is in Paris, France."
    assert retriever._calculate_relevance(query_terms, irrelevant_doc) == 0
