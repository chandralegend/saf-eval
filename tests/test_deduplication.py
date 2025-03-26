import pytest
import uuid

from saf_eval.core.models import AtomicFact
from saf_eval.utils.deduplication import deduplicate_facts, _calculate_similarity

@pytest.fixture
def duplicate_facts():
    """Create a list of facts with some duplicates."""
    return [
        AtomicFact(
            id=str(uuid.uuid4()),
            text="The Eiffel Tower is located in Paris, France.",
            source_text="Sample source text"
        ),
        AtomicFact(
            id=str(uuid.uuid4()),
            text="The Eiffel Tower is in Paris, France.",  # Similar to the first one
            source_text="Sample source text"
        ),
        AtomicFact(
            id=str(uuid.uuid4()),
            text="The Louvre Museum houses the Mona Lisa.",
            source_text="Sample source text"
        ),
        AtomicFact(
            id=str(uuid.uuid4()),
            text="The Mona Lisa is displayed in the Louvre Museum.",  # Similar to the third one
            source_text="Sample source text"
        ),
        AtomicFact(
            id=str(uuid.uuid4()),
            text="Rome is the capital city of Italy.",  # Unique fact
            source_text="Sample source text"
        )
    ]

def test_calculate_similarity():
    """Test the similarity calculation function."""
    # Identical texts
    assert _calculate_similarity("This is a test", "This is a test") == 1.0
    
    # Similar texts
    high_similarity = _calculate_similarity(
        "The Eiffel Tower is in Paris", 
        "The Eiffel Tower is located in Paris"
    )
    assert high_similarity > 0.8
    
    # Different texts
    low_similarity = _calculate_similarity(
        "The Eiffel Tower is in Paris", 
        "The Louvre Museum is in Paris"
    )
    assert low_similarity < 0.7

def test_deduplicate_facts(duplicate_facts):
    """Test the deduplication of atomic facts."""
    # With default threshold
    deduplicated = deduplicate_facts(duplicate_facts)
    
    # Should eliminate 2 duplicates
    assert len(deduplicated) == 4
    
    # Check which facts were kept
    texts = [fact.text for fact in deduplicated]
    assert "The Eiffel Tower is located in Paris, France." in texts
    assert "The Louvre Museum houses the Mona Lisa." in texts
    assert "Rome is the capital city of Italy." in texts
    
    # With a lower threshold, more facts should be considered duplicates
    deduplicated_strict = deduplicate_facts(duplicate_facts, similarity_threshold=0.4)
    assert len(deduplicated_strict) <= 3
    
    # With a very high threshold, no facts should be considered duplicates
    deduplicated_loose = deduplicate_facts(duplicate_facts, similarity_threshold=0.99)
    assert len(deduplicated_loose) == 5

def test_custom_deduplication_function():
    """Test using a custom deduplication function."""
    facts = [
        AtomicFact(
            id=str(uuid.uuid4()),
            text="Fact 1",
            source_text="Sample source text"
        ),
        AtomicFact(
            id=str(uuid.uuid4()),
            text="Fact 2",
            source_text="Sample source text"
        ),
        AtomicFact(
            id=str(uuid.uuid4()),
            text="Fact 3",
            source_text="Sample source text"
        )
    ]
    
    # Custom function that keeps only facts with odd indices
    def custom_dedupe(facts_list):
        return [fact for i, fact in enumerate(facts_list) if i % 2 == 0]
    
    result = custom_dedupe(facts)
    assert len(result) == 2
    assert result[0].text == "Fact 1"
    assert result[1].text == "Fact 3"
