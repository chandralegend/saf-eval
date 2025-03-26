from typing import Dict, List, Set
from pydantic import BaseSettings, Field

class Config(BaseSettings):
    """Global configuration for SAF-Eval pipeline.
    
    This class manages the configuration for the entire evaluation pipeline.
    The scoring_rubric defines both the evaluation categories and their weights.
    """
    # Required configuration settings (should not be changed after initialization)
    retrieval_method: str = Field(default="default", description="Method used for document retrieval")
    scoring_rubric: Dict[str, float] = Field(
        default={"relevant": 1.0, "irrelevant": 0.0}, 
        description="Scoring rubric for fact classifications. The keys also define the valid evaluation categories."
    )
    
    # Optional/extensible configuration
    llm_config: Dict = Field(default_factory=dict, description="Configuration for LLM providers")
    retrieval_config: Dict = Field(default_factory=dict, description="Configuration for retrieval methods")
    
    @property
    def evaluation_categories(self) -> List[str]:
        """Get valid evaluation categories from the scoring rubric."""
        return list(self.scoring_rubric.keys())
    
    class Config:
        env_prefix = "SAFEVAL_"
        protected_namespaces = ("scoring_rubric", "evaluation_categories")
