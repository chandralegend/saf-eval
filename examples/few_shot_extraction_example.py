import asyncio
import os
from typing import List, Optional, Tuple, Any
from dotenv import load_dotenv

from saf_eval.config import Config
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.core.models import AtomicFact
from saf_eval.llm.providers.openai import OpenAILLM

# Load environment variables
load_dotenv()

# Example provider function that returns examples based on the domain of the text
def domain_specific_examples(response: str, context: Optional[str] = None, **kwargs) -> List[Tuple[str, List[str]]]:
    """
    Provide domain-specific examples for fact extraction.
    
    Analyzes the response and provides relevant examples based on content domain.
    """
    # Detect domain from response and context
    text_to_analyze = (response + " " + (context or "")).lower()
    
    # Examples for different domains
    if any(word in text_to_analyze for word in ["science", "physics", "chemistry", "biology"]):
        # Science domain examples
        return [
            (
                "The Human Genome Project was completed in 2003, mapping approximately 92% of the human genome. "
                "It cost around $3 billion and involved scientists from 20 institutions across 6 countries.",
                [
                    "The Human Genome Project was completed in 2003.",
                    "The Human Genome Project mapped approximately 92% of the human genome.",
                    "The Human Genome Project cost around $3 billion.",
                    "The Human Genome Project involved scientists from 20 institutions.",
                    "The Human Genome Project involved scientists from 6 countries."
                ]
            ),
            (
                "Water boils at 100°C at sea level, but at higher altitudes it boils at lower temperatures.",
                [
                    "Water boils at 100°C at sea level.",
                    "At higher altitudes, water boils at lower temperatures."
                ]
            )
        ]
    elif any(word in text_to_analyze for word in ["history", "war", "century", "ancient"]):
        # History domain examples
        return [
            (
                "World War II ended in 1945 with the surrender of Japan after the United States dropped atomic bombs "
                "on Hiroshima and Nagasaki. The war lasted six years and involved more than 30 countries.",
                [
                    "World War II ended in 1945.",
                    "Japan surrendered at the end of World War II.",
                    "The United States dropped atomic bombs on Hiroshima and Nagasaki.",
                    "World War II lasted six years.",
                    "World War II involved more than 30 countries."
                ]
            ),
            (
                "The Roman Empire reached its greatest territorial extent under Emperor Trajan in 117 CE, covering 5 million square kilometers.",
                [
                    "The Roman Empire reached its greatest territorial extent under Emperor Trajan.",
                    "The Roman Empire's maximum extent occurred in 117 CE.",
                    "At its height, the Roman Empire covered 5 million square kilometers."
                ]
            )
        ]
    else:
        # General domain examples
        return [
            (
                "Paris is the capital of France and has a population of 2.1 million people. The Eiffel Tower, "
                "completed in 1889, stands at 330 meters tall.",
                [
                    "Paris is the capital of France.",
                    "Paris has a population of 2.1 million people.",
                    "The Eiffel Tower was completed in 1889.",
                    "The Eiffel Tower is 330 meters tall."
                ]
            )
        ]

async def main():
    # Initialize configuration
    config = Config()
    
    # Initialize LLM
    llm = OpenAILLM(
        model="gpt-3.5-turbo", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create extractors with and without examples
    standard_extractor = FactExtractor(config=config, llm=llm)
    example_based_extractor = FactExtractor(
        config=config, 
        llm=llm,
        example_provider=domain_specific_examples
    )
    
    # Test responses from different domains
    science_response = """
    Quantum computing leverages quantum mechanics to process information in ways classical computers cannot.
    IBM's quantum computers use superconducting qubits that operate at near absolute zero temperature.
    Recent advancements have allowed quantum systems to achieve quantum advantage in specific tasks.
    """
    
    history_response = """
    The Byzantine Empire, also known as the Eastern Roman Empire, survived for a thousand years after the fall of Rome.
    Constantinople was its capital until it fell to the Ottoman Turks in 1453.
    Emperor Justinian I, who ruled from 527 to 565, is known for his ambitious building projects and legal reforms.
    """
    
    # Extract facts with and without examples
    print("=== Science Domain ===")
    print("\nStandard extraction:")
    standard_science_facts = await standard_extractor.extract_facts(science_response)
    for i, fact in enumerate(standard_science_facts):
        print(f"{i+1}. {fact.text}")
        
    print("\nExample-based extraction:")
    example_science_facts = await example_based_extractor.extract_facts(science_response)
    for i, fact in enumerate(example_science_facts):
        print(f"{i+1}. {fact.text}")
    
    print("\n\n=== History Domain ===")
    print("\nStandard extraction:")
    standard_history_facts = await standard_extractor.extract_facts(history_response)
    for i, fact in enumerate(standard_history_facts):
        print(f"{i+1}. {fact.text}")
        
    print("\nExample-based extraction:")
    example_history_facts = await example_based_extractor.extract_facts(history_response)
    for i, fact in enumerate(example_history_facts):
        print(f"{i+1}. {fact.text}")
    
    # Compare number of facts extracted and detail level
    print("\n\n=== Comparison ===")
    print(f"Science domain - Standard extraction: {len(standard_science_facts)} facts")
    print(f"Science domain - Example-based extraction: {len(example_science_facts)} facts")
    print(f"History domain - Standard extraction: {len(standard_history_facts)} facts")
    print(f"History domain - Example-based extraction: {len(example_history_facts)} facts")

if __name__ == "__main__":
    asyncio.run(main())
