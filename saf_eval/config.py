from typing import Dict, List, Optional
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings

class LoggingConfig(BaseSettings):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    console: bool = Field(default=True, description="Whether to log to console")
    file: bool = Field(default=False, description="Whether to log to file")
    log_dir: Optional[str] = Field(default=None, description="Directory for log files")
    json_format: bool = Field(default=False, description="Whether to format logs as JSON")

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
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    
    @property
    def evaluation_categories(self) -> List[str]:
        """Get valid evaluation categories from the scoring rubric."""
        return list(self.scoring_rubric.keys())
    
    # Using ConfigDict instead of class Config
    model_config = ConfigDict(
        env_prefix="SAFEVAL_",
        protected_namespaces=("evaluation_categories",)
    )
