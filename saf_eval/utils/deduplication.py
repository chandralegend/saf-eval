from typing import List, Callable, Optional
from difflib import SequenceMatcher

from ..core.models import AtomicFact

def deduplicate_facts(facts: List[AtomicFact], similarity_threshold: float = 0.85) -> List[AtomicFact]:
    """
    Deduplicate atomic facts by removing highly similar facts.
    
    Args:
        facts: List of atomic facts to deduplicate
        similarity_threshold: Threshold above which facts are considered duplicates (0.0-1.0)
        
    Returns:
        List of deduplicated facts
    """
    if not facts:
        return []
    
    # Start with the first fact
    unique_facts = [facts[0]]
    
    # Compare each fact with the ones we've already kept
    for fact in facts[1:]:
        is_duplicate = False
        for unique_fact in unique_facts:
            similarity = _calculate_similarity(fact.text, unique_fact.text)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_facts.append(fact)
    
    return unique_facts

def _calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
