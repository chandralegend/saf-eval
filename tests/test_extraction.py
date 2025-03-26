import pytest
import uuid
from unittest.mock import AsyncMock, patch

from saf_eval.core.models import AtomicFact
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.llm.base import LLMBase

@pytest.fixture
def mock_llm():
    mock = AsyncMock(spec=LLMBase)
    mock.generate.return_value = "Fact 1\nFact 2\nFact 3"
    return mock

@pytest.fixture
def extractor(mock_llm):
    return FactExtractor(llm=mock_llm)

@pytest.fixture
def basic_extractor():
    return FactExtractor()

def test_extract_basic(basic_extractor):
    response = "This is fact one. This is fact two. This is fact three."
    facts = basic_extractor._extract_basic(response)
    
    assert len(facts) == 3
    assert all(isinstance(fact, AtomicFact) for fact in facts)
    assert facts[0].text == "This is fact one"
    assert facts[1].text == "This is fact two"
    assert facts[2].text == "This is fact three"
    assert all(fact.source_text == response for fact in facts)

@pytest.mark.asyncio
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

@pytest.mark.asyncio
async def test_extract_facts_with_llm(extractor):
    response = "Test response"
    facts = await extractor.extract_facts(response)
    
    assert len(facts) == 3
    assert all(isinstance(fact, AtomicFact) for fact in facts)

@pytest.mark.asyncio
async def test_extract_facts_basic(basic_extractor):
    response = "First fact. Second fact."
    facts = await basic_extractor.extract_facts(response)
    
    assert len(facts) == 2
    assert facts[0].text == "First fact"
    assert facts[1].text == "Second fact"
