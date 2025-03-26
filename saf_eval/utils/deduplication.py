from typing import List, Callable, Optional
from difflib import SequenceMatcher
import logging

from ..core.models import AtomicFact

# Get the default logger
logger = logging.getLogger("saf-eval")

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
    
    logger.debug(f"Starting deduplication of {len(facts)} facts with threshold {similarity_threshold}")
    
    # Start with the first fact
    unique_facts = [facts[0]]
    duplicates = []
    
    # Compare each fact with the ones we've already kept
    for fact in facts[1:]:
        is_duplicate = False
        duplicate_of = None
        highest_similarity = 0
        
        for unique_fact in unique_facts:
            similarity = _calculate_similarity(fact.text, unique_fact.text)
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                duplicate_of = unique_fact.text
            
            if similarity >= similarity_threshold:
                is_duplicate = True
                duplicates.append((fact.text, unique_fact.text, similarity))
                logger.debug(f"Found duplicate fact: '{fact.text}' similar to '{unique_fact.text}' (score: {similarity:.2f})")
                break
        
        if not is_duplicate:
            unique_facts.append(fact)
        else:
            logger.debug(f"Skipping duplicate fact: '{fact.text}' (similar to: '{duplicate_of}')")
    
    logger.info(f"Deduplication complete: {len(unique_facts)} unique facts, {len(duplicates)} duplicates removed")
    
    return unique_facts

def _calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
