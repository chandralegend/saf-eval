# SAF-Eval

[![Run Tests](https://github.com/chandralegend/saf-eval/actions/workflows/test.yml/badge.svg)](https://github.com/chandralegend/saf-eval/actions/workflows/test.yml)
[![Lint](https://github.com/chandralegend/saf-eval/actions/workflows/lint.yml/badge.svg)](https://github.com/chandralegend/saf-eval/actions/workflows/lint.yml)

SAF-Eval (Search-Augmented Factuality Evaluator) is a modular Python package for evaluating the factuality of AI-generated responses. Based on academic research, it implements a systematic approach to measuring factual accuracy by breaking down responses into atomic facts and evaluating them against retrieved evidence.

## Features

- **Modular Pipeline**: Extract atomic facts, check relevance, retrieve supporting documents, evaluate factuality
- **Self-Containment Processing**: Automatically detect and fix non-self-contained facts by adding context
- **Few-Shot Learning**: Improve fact extraction using domain-specific examples
- **Fact Deduplication**: Identify and remove similar or duplicate facts to avoid redundant evaluations
- **Comprehensive Logging**: Detailed logging of the entire evaluation process for analysis and debugging
- **Customizable Evaluation**: Define your own categories and scoring rubrics
- **Provider-Agnostic**: Use any LLM provider through a consistent interface
- **Flexible Retrieval**: Integrate with any document retrieval system
- **Comprehensive Metrics**: Get detailed factuality scores and evaluations

## Installation

SAF-Eval requires Python 3.12 or later.

### Using Poetry (recommended)

```bash
# Clone the repository
git clone https://github.com/chandralegend/saf-eval.git
cd saf-eval

# Install dependencies with Poetry
poetry install
```

### Using pip

```bash
pip install saf-eval
```

## Quick Start

```python
import asyncio
import os
from dotenv import load_dotenv

from saf_eval.config import Config, LoggingConfig
from saf_eval.core.pipeline import EvaluationPipeline
from saf_eval.extraction.extractor import FactExtractor
from saf_eval.containment.checker import ContainmentChecker
from saf_eval.llm.providers.openai import OpenAILLM
from saf_eval.retrieval.providers.simple import SimpleRetriever
from saf_eval.evaluation.classifier import FactClassifier
from saf_eval.evaluation.scoring import FactualityScorer

# Load environment variables (for API keys)
load_dotenv()

async def evaluate_response():
    # Initialize components with shared config
    config = Config(
        scoring_rubric={
            "supported": 1.0,
            "contradicted": 0.0, 
            "unverifiable": 0.5
        },
        retrieval_config={"top_k": 3},
        logging=LoggingConfig(level="INFO", console=True, file=True, log_dir="./logs")
    )
    
    llm = OpenAILLM(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create a knowledge base
    knowledge_base = {
        "Mount Everest": "Mount Everest is the highest mountain above sea level at 29,032 feet (8,849 meters)."
    }
    
    # Setup the pipeline with components that share the same config
    pipeline = EvaluationPipeline(
        config=config,
        extractor=FactExtractor(config=config, llm=llm),
        retriever=SimpleRetriever(config=config, knowledge_base=knowledge_base),
        classifier=FactClassifier(config=config, llm=llm),
        scorer=FactualityScorer(config=config),
        containment_checker=ContainmentChecker(config=config, llm=llm)  # Add containment checker
    )
    
    # Evaluate a response
    response = "Mount Everest, at 29,032 feet, is the tallest mountain on Earth."
    context = "Information about geographical features"
    
    result = await pipeline.run(response, context)
    print(f"Factuality Score: {result.factuality_score:.2f}")

if __name__ == "__main__":
    asyncio.run(evaluate_response())
```

## Project Structure

SAF-Eval follows a modular architecture with the following key components:

- **Core**: Pipeline coordination and data models
- **Extraction**: Breaking down responses into atomic facts
- **Containment**: Checking and fixing non-self-contained facts
- **Relevancy**: Assessing relevance of facts to the context
- **Retrieval**: Finding supporting documents for verification
- **Evaluation**: Classifying facts and calculating factuality scores
- **LLM**: Abstraction layer for language model providers
- **Utils**: Utility functions including deduplication and logging

## Advanced Usage

### Self-Containment Processing

The `ContainmentChecker` helps identify and fix facts that require additional context to be understood:

```python
from saf_eval.containment.checker import ContainmentChecker
from saf_eval.llm.providers.openai import OpenAILLM

# Initialize the components
llm = OpenAILLM(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
containment_checker = ContainmentChecker(config=config, llm=llm)

# Check if facts are self-contained
checked_facts = await containment_checker.check_containment(facts, response)

# Fix non-self-contained facts by adding context
self_contained_facts = await containment_checker.self_contain_facts(
    checked_facts, 
    response, 
    context="Optional additional context to help with self-containment"
)

# Example: Converting "He wrote it in 1851" to "Herman Melville wrote Moby Dick in 1851"
```

### Using Context in Fact Extraction

The fact extractor can now use context to improve extraction quality:

```python
extractor = FactExtractor(config=config, llm=llm)
facts = await extractor.extract_facts(
    response="Melville's masterpiece is considered one of the Great American Novels.",
    context="Discussion about the novel Moby Dick by Herman Melville"
)
```

### Using Examples for Fact Extraction

You can provide examples to improve fact extraction through few-shot learning:

```python
from typing import List, Tuple, Optional

# Define an example provider function
def my_example_provider(response: str, context: Optional[str] = None, **kwargs) -> List[Tuple[str, List[str]]]:
    """Provide domain-specific examples for fact extraction."""
    # Return a list of (example_text, [fact1, fact2, ...]) tuples
    return [
        (
            "The Eiffel Tower was completed in 1889 and stands at 330 meters tall.",
            ["The Eiffel Tower was completed in 1889.", "The Eiffel Tower is 330 meters tall."]
        ),
        # More examples...
    ]

# Use the example provider in the extractor
extractor = FactExtractor(
    config=config,
    llm=llm,
    example_provider=my_example_provider
)

# Extract facts with examples to guide the LLM
facts = await extractor.extract_facts(
    response="The Golden Gate Bridge was completed in 1937.",
    context="Information about famous structures"
)
```

Example providers can dramatically improve extraction quality by demonstrating the desired level of granularity and format.

### Custom Retrieval System

You can implement your own retrieval system:

```python
from typing import List
from saf_eval.core.models import AtomicFact, RetrievedDocument
from saf_eval.retrieval.base import RetrieverBase

class MyCustomRetriever(RetrieverBase):
    async def retrieve(self, fact: AtomicFact, **kwargs) -> List<RetrievedDocument]:
        # Implement your retrieval logic here
        # ...
        return documents
```

### Custom Evaluation Categories

Customize how facts are classified:

```python
from saf_eval.evaluation.classifier import FactClassifier

classifier = FactClassifier(
    llm=my_llm,
    categories=["accurate", "partially_accurate", "inaccurate", "uncertain"]
)
```

### Fact Deduplication

The pipeline automatically deduplicates similar facts to avoid redundant evaluations:

```python
from saf_eval.utils.deduplication import deduplicate_facts
from typing import List
from saf_eval.core.models import AtomicFact

# Default deduplication is already included in the pipeline, but can be customized:
def my_custom_deduplication(facts: List<AtomicFact]) -> List<AtomicFact]:
    # Custom logic to identify and merge similar facts
    # ...
    return deduplicated_facts

# Use custom deduplication in the pipeline
pipeline = EvaluationPipeline(
    # ...other arguments...
    deduplication_fn=my_custom_deduplication
)
```

### Comprehensive Logging

SAF-Eval provides detailed logging throughout the evaluation pipeline:

```python
from saf_eval.config import Config, LoggingConfig
from saf_eval.utils.logging import get_logger

# Configure logging in the config object
config = Config(
    # ...other config settings...
    logging=LoggingConfig(
        level="DEBUG",           # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console=True,            # Log to console
        file=True,               # Log to file
        log_dir="./logs",        # Directory for log files
        json_format=True         # Output logs in JSON format for easier parsing
    )
)

# Create a custom logger in any component
logger = get_logger(
    name="my-component",
    level=config.logging.level,
    log_dir=config.logging.log_dir,
    console=config.logging.console,
    file=config.logging.file,
    json_format=config.logging.json_format
)

# Log with structured data
logger.info("Processing fact", {
    "fact_id": "123",
    "fact_text": "Paris is the capital of France",
    "is_self_contained": True
})
```

The logging system tracks:
- Fact extraction steps and results
- Self-containment checks and fixes
- Deduplication decisions
- Document retrieval statistics
- Classification results
- Overall performance metrics

For a complete example, see `examples/logging_example.py`.

## Configuration

SAF-Eval uses a unified configuration system. The `Config` class centralizes all settings:

```python
from saf_eval.config import Config

config = Config(
    # The scoring rubric defines both evaluation categories and their weights
    scoring_rubric={
        "fully_supported": 1.0,
        "partially_supported": 0.6,
        "contradicted": 0.0
    },
    # Additional configuration for retrieval methods
    retrieval_config={
        "top_k": 5,
        "min_relevance": 0.7
    },
    # Custom LLM configuration
    llm_config={
        "temperature": 0.1,
        "max_tokens": 500
    }
)

# Categories are automatically derived from the scoring rubric
print(config.evaluation_categories)  # ['fully_supported', 'partially_supported', 'contradicted']
```

See the `examples/` directory for more advanced usage patterns, including `self_containment_example.py` which demonstrates how to process non-self-contained facts.

## Example Scripts

SAF-Eval includes several example scripts to demonstrate key features:

- **basic_usage.py**: Simple end-to-end evaluation
- **custom_retriever.py**: Implementing a custom retrieval system
- **custom_evaluation.py**: Custom evaluation categories and scoring
- **self_containment_example.py**: Handling non-self-contained facts
- **deduplication_example.py**: Custom fact deduplication
- **few_shot_extraction_example.py**: Domain-specific extraction examples
- **logging_example.py**: Comprehensive logging setup

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
