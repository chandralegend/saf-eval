# SAF-Eval

[![Run Tests](https://github.com/chandralegend/saf-eval/actions/workflows/test.yml/badge.svg)](https://github.com/chandralegend/saf-eval/actions/workflows/test.yml)
[![Lint](https://github.com/chandralegend/saf-eval/actions/workflows/lint.yml/badge.svg)](https://github.com/chandralegend/saf-eval/actions/workflows/lint.yml)

SAF-Eval (Search-Augmented Factuality Evaluator) is a modular Python package for evaluating the factuality of AI-generated responses. Based on academic research, it implements a systematic approach to measuring factual accuracy by breaking down responses into atomic facts and evaluating them against retrieved evidence.

## Features

- **Modular Pipeline**: Extract atomic facts, check relevance, retrieve supporting documents, evaluate factuality
- **Self-Containment Processing**: Automatically detect and fix non-self-contained facts by adding context
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

from saf_eval.config import Config
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
        retrieval_config={"top_k": 3}
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
def my_custom_deduplication(facts: List[AtomicFact]) -> List[AtomicFact]:
    # Custom logic to identify and merge similar facts
    # ...
    return deduplicated_facts

# Use custom deduplication in the pipeline
pipeline = EvaluationPipeline(
    # ...other arguments...
    deduplication_fn=my_custom_deduplication
)
```

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
