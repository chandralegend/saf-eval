import asyncio
import os
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from saf_eval.config import Config
from saf_eval.llm.providers.openai import OpenAILLM

# Load environment variables
load_dotenv()

# Define Pydantic models for structured output
class Step(BaseModel):
    """A step in a reasoning process."""
    explanation: str
    output: str

class MathReasoning(BaseModel):
    """A mathematical reasoning process with steps and final answer."""
    steps: List[Step]
    final_answer: str

class FactAnalysis(BaseModel):
    """Analysis of a factual statement."""
    is_factual: bool
    # Remove the constraints that cause the API error
    confidence: float  # OpenAI beta API doesn't support Field(ge=0.0, le=1.0)
    reasoning: str
    sources_needed: Optional[List[str]] = None

async def main():
    # Initialize configuration
    config = Config()
    
    # Initialize LLM
    llm = OpenAILLM(
        model="gpt-4o-2024-08-06",  # Use a model that supports parsing
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print("=== Math Problem Solving with Structured Output ===")
    # Example 1: Math reasoning
    math_problem = "How can I solve 8x + 7 = -23?"
    
    math_reasoning = await llm.generate_structured(
        prompt=math_problem,
        schema_model=MathReasoning,
        system_prompt="You are a helpful math tutor. Guide the user through the solution step by step."
    )
    
    # Display the structured output
    print(f"\nSolving: {math_problem}")
    print("\nSolution steps:")
    for i, step in enumerate(math_reasoning.steps, 1):
        print(f"\nStep {i}:")
        print(f"Explanation: {step.explanation}")
        print(f"Output: {step.output}")
    
    print(f"\nFinal answer: {math_reasoning.final_answer}")
    
    # Example 2: Fact analysis
    print("\n\n=== Fact Analysis with Structured Output ===")
    fact = "The Great Wall of China is visible from the Moon."
    
    # Add instruction to keep confidence between 0-1 in the system prompt instead
    system_prompt = """
    You are a factuality expert. Analyze the given statement and determine if it's factual.
    Express your confidence as a number between 0.0 and 1.0, where 0.0 is completely unconfident 
    and 1.0 is completely confident.
    """
    
    fact_analysis = await llm.generate_structured(
        prompt=f"Analyze this statement: '{fact}'",
        schema_model=FactAnalysis,
        system_prompt=system_prompt
    )
    
    # Display the structured output
    print(f"\nAnalyzing: {fact}")
    print(f"\nIs factual: {fact_analysis.is_factual}")
    print(f"Confidence: {fact_analysis.confidence:.2f}")
    print(f"Reasoning: {fact_analysis.reasoning}")
    
    if fact_analysis.sources_needed:
        print("\nSources needed:")
        for source in fact_analysis.sources_needed:
            print(f"- {source}")

if __name__ == "__main__":
    asyncio.run(main())
