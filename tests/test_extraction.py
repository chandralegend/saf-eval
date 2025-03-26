import pytest
import uuid
from unittest.mock import AsyncMock

from saf_eval.core.models import AtomicFact
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.llm.base import LLMBase

# config fixture now comes from conftest.py

@pytest.fixture
def mock_llm():
    mock = AsyncMock(spec=LLMBase)
    mock.generate.return_value = "Fact 1\nFact 2\nFact 3"
    return mock

@pytest.fixture
def extractor(config, mock_llm):
    return FactExtractor(config=config, llm=mock_llm)

@pytest.fixture
def basic_extractor(config):
    return FactExtractor(config=config)

@pytest.fixture
def example_provider():
    def provide_examples(response, context=None, **kwargs):
        return [
            (
                "The moon orbits Earth at an average distance of 384,400 km.",
                ["The moon orbits Earth.", "The average distance between the moon and Earth is 384,400 km."]
            )
        ]
    return provide_examples

@pytest.fixture
def example_extractor(config, mock_llm, example_provider):
    return FactExtractor(config=config, llm=mock_llm, example_provider=example_provider)

def test_extract_basic(basic_extractor):
    response = "This is fact one. This is fact two. This is fact three."
    facts = basic_extractor._extract_basic(response)
    
    assert len(facts) == 3
    assert all(isinstance(fact, AtomicFact) for fact in facts)
    assert facts[0].text == "This is fact one"
    assert facts[1].text == "This is fact two"
    assert facts[2].text == "This is fact three"
    assert all(fact.source_text == response for fact in facts)

# pytest-asyncio will now recognize these tests automatically
async def test_extract_with_llm(extractor, mock_llm):
    response = "This is a response with multiple facts."
    facts = await extractor._extract_with_llm(response)
    
    assert len(facts) == 3
    assert all(isinstance(fact, AtomicFact) for fact in facts)
    assert facts[0].text == "Fact 1"
    assert facts[1].text == "Fact 2"
    assert facts[2].text == "Fact 3"
    assert all(fact.source_text == response for fact in facts)
    
    mock_llm.generate.assert_called_once()
    assert response in mock_llm.generate.call_args[0][0]

async def test_extract_facts_with_llm(extractor):
    response = "Test response"
    facts = await extractor.extract_facts(response)
    
    assert len(facts) == 3
    assert all(isinstance(fact, AtomicFact) for fact in facts)

async def test_extract_facts_basic(basic_extractor):
    response = "First fact. Second fact."
    facts = await basic_extractor.extract_facts(response)
    
    assert len(facts) == 2
    assert facts[0].text == "First fact"
    assert facts[1].text == "Second fact"

async def test_extract_with_examples(example_extractor, mock_llm, example_provider):
    response = "Mars has two moons, Phobos and Deimos."
    
    # Configure mock to return a specific response for the example-based prompt
    mock_llm.generate.return_value = "- Mars has two moons.\n- Phobos is a moon of Mars.\n- Deimos is a moon of Mars."
    
    facts = await example_extractor.extract_facts(response)
    
    assert len(facts) == 3
    assert "Mars has two moons" in facts[0].text
    assert "Phobos" in facts[1].text
    assert "Deimos" in facts[2].text
    
    # Verify that the examples were included in the prompt
    call_args = mock_llm.generate.call_args[0][0]
    assert "The moon orbits Earth" in call_args
    assert "Mars has two moons" in call_args
