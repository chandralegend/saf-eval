from typing import List, Optional, Callable, Tuple, Any
import uuid

from ..config import Config
from ..core.models import AtomicFact
from ..llm.base import LLMBase
from ..utils.logging import get_logger

# Type definition for example provider function
ExampleProviderType = Callable[[str, Optional[str], Any], List[Tuple[str, List[str]]]]

class FactExtractor:
    """Extract atomic facts from an AI response."""
    
    def __init__(self, config: Config, llm: LLMBase = None, 
                 example_provider: Optional[ExampleProviderType] = None):
        """
        Initialize the fact extractor.
        
        Args:
            config: Configuration object
            llm: Optional LLM implementation for extraction
            example_provider: Optional function that provides examples for few-shot learning
                              Should return List[Tuple[str, List[str]]] where each tuple is 
                              (example_text, [fact1, fact2, ...])
        """
        self.config = config
        self.llm = llm
        self.example_provider = example_provider
        self.logger = get_logger(
            name="extractor",
            level=self.config.logging.level,
            log_dir=self.config.logging.log_dir,
            console=self.config.logging.console,
            file=self.config.logging.file,
            json_format=self.config.logging.json_format
        )
        
    async def extract_facts(self, response: str, context: Optional[str] = None, **kwargs) -> List[AtomicFact]:
        """
        Extract atomic facts from the response.
        
        Args:
            response: The text response to extract facts from
            context: Optional context to help with extraction (used when LLM is available)
            **kwargs: Additional arguments passed to example_provider if set
        
        Returns:
            List of extracted atomic facts
        """
        self.logger.info(f"Extracting facts from response (length: {len(response)})")
        self.logger.debug(f"Context provided: {context is not None}")
        
        if self.llm:
            result = await self._extract_with_llm(response, context, **kwargs)
            self.logger.info(f"Extracted {len(result)} facts using LLM")
            return result
        else:
            result = self._extract_basic(response)
            self.logger.info(f"Extracted {len(result)} facts using basic extraction")
            return result
            
    async def _extract_with_llm(self, response: str, context: Optional[str] = None, **kwargs) -> List[AtomicFact]:
        """Use LLM to extract atomic facts."""
        # Build the base prompt
        prompt = """
        Extract the atomic facts from the following text. An atomic fact is a simple, 
        self-contained statement that makes a single factual claim.
        """
        
        # Add examples if an example provider is set
        example_count = 0
        if self.example_provider:
            self.logger.debug("Using example provider for few-shot extraction")
            examples = self.example_provider(response, context, **kwargs)
            if examples:
                example_count = len(examples)
                self.logger.debug(f"Adding {example_count} examples for few-shot learning")
                prompt += "\n\nHere are some examples of atomic fact extraction:"
                
                for i, (example_text, example_facts) in enumerate(examples):
                    prompt += f"\n\nExample {i+1}:\nText: {example_text}\n"
                    prompt += "Atomic facts:\n" + "\n".join([f"- {fact}" for fact in example_facts])
        else:
            self.logger.debug("No example provider, using zero-shot extraction")
        
        # Add the actual response to analyze
        prompt += f"""

        Now extract atomic facts from this text:
        
        Text: {response}
        """
        
        if context:
            prompt += f"""
            
            Context (to help understand the text):
            {context}
            """
        
        prompt += "\n\nAtomic facts (one per line):"
        
        self.logger.debug("Sending extraction prompt to LLM")
        result = await self.llm.generate(prompt)
        
        # Clean up the result to handle potential bullet points or numbering
        cleaned_result = result.strip()
        # Remove bullet points, numbers, or dashes at the beginning of lines
        lines = cleaned_result.split('\n')
        facts = []
        for line in lines:
            line = line.strip()
            # Remove common list prefixes
            line = line.lstrip('•').lstrip('-').lstrip('*').strip()
            # Remove numbering like "1.", "2.", etc.
            if line and len(line) > 2 and line[0].isdigit() and line[1:].startswith('. '):
                line = line[line.find('.')+1:].strip()
            if line:
                facts.append(line)
        
        atomic_facts = [
            AtomicFact(
                id=str(uuid.uuid4()),
                text=fact.strip(),
                source_text=response
            )
            for fact in facts if fact.strip()
        ]
        
        self.logger.debug(f"Extracted {len(atomic_facts)} facts from response")
        for i, fact in enumerate(atomic_facts):
            self.logger.debug(f"Fact {i+1}: {fact.text}", {"fact_index": i, "fact_text": fact.text})
        
        return atomic_facts
    
    def _extract_basic(self, response: str) -> List[AtomicFact]:
        """Basic extraction by splitting on periods."""
        self.logger.debug("Using basic extraction by splitting on periods")
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        atomic_facts = [
            AtomicFact(
                id=str(uuid.uuid4()),
                text=sentence,
                source_text=response
            )
            for sentence in sentences
        ]
        
        self.logger.debug(f"Extracted {len(atomic_facts)} facts by sentence splitting")
        return atomic_facts
