import pytest
import uuid
from unittest.mock import AsyncMock

from saf_eval.core.models import AtomicFact
from saf_eval.containment.checker import ContainmentChecker
from saf_eval.llm.base import LLMBase

@pytest.fixture
def mock_llm():
    mock = AsyncMock(spec=LLMBase)
    # More explicit response handling to ensure correct test behavior
    mock.generate.side_effect = lambda prompt, **kwargs: _mock_generate_response(prompt)
    return mock

def _mock_generate_response(prompt):
    """Helper to provide appropriate responses based on prompt content"""
    if "Is this fact self-contained?" in prompt:
        if "Fact to check: Moby Dick is a famous novel." in prompt:
            return "yes"
        elif "Fact to check: He wrote it in the mid-19th century." in prompt:
            return "no"
        else:
            return "unknown response"
    elif "Rewrite this as a self-contained fact:" in prompt:
        if "Fact to make self-contained: He wrote it in the mid-19th century." in prompt:
            return "Herman Melville wrote Moby Dick in the mid-19th century."
        else:
            return "unknown self-contained fact"
    else:
        return "Unexpected prompt"

@pytest.fixture
def containment_checker(config, mock_llm):
    return ContainmentChecker(config=config, llm=mock_llm)

@pytest.fixture
def atomic_facts():
    # Ensure we return exactly two items with specific texts
    return [
        AtomicFact(
            id=str(uuid.uuid4()),
            text="Moby Dick is a famous novel.",
            source_text="Moby Dick is a famous novel by Herman Melville."
        ),
        AtomicFact(
            id=str(uuid.uuid4()),
            text="He wrote it in the mid-19th century.",
            source_text="Moby Dick is a famous novel by Herman Melville. He wrote it in the mid-19th century."
        )
    ]

async def test_check_containment(containment_checker, atomic_facts, mock_llm):
    response = "Moby Dick is a famous novel by Herman Melville. He wrote it in the mid-19th century."
    
    # Verify we have exactly two facts with expected content
    assert len(atomic_facts) == 2
    assert atomic_facts[0].text == "Moby Dick is a famous novel."
    assert atomic_facts[1].text == "He wrote it in the mid-19th century."
    
    result = await containment_checker.check_containment(atomic_facts, response)
    
    assert len(result) == 2
    assert result[0].is_self_contained == True
    assert result[1].is_self_contained == False

async def test_self_contain_facts(containment_checker, atomic_facts, mock_llm):
    response = "Moby Dick is a famous novel by Herman Melville. He wrote it in the mid-19th century."
    context = "Information about literary works"
    
    # Make sure we have both facts and set their self-contained status
    assert len(atomic_facts) == 2, "Test requires exactly 2 atomic facts"
    atomic_facts[0].is_self_contained = True
    atomic_facts[1].is_self_contained = False
    
    result = await containment_checker.self_contain_facts(atomic_facts, response, context)
    
    assert len(result) == 2
    # First fact should remain unchanged since it's already self-contained
    assert result[0].text == "Moby Dick is a famous novel."
    assert result[0].is_self_contained == True
    
    # Second fact should be updated to be self-contained
    assert result[1].text == "Herman Melville wrote Moby Dick in the mid-19th century."
    assert result[1].is_self_contained == True
    
    # Verify that LLM generate was called with appropriate prompts
    generate_calls = [call[0][0] for call in mock_llm.generate.call_args_list if "Rewrite this as a self-contained fact:" in call[0][0]]
    assert len(generate_calls) > 0
    assert "He wrote it in the mid-19th century" in generate_calls[0]
    assert "Information about literary works" in generate_calls[0]
