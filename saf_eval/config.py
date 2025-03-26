from pydantic import BaseSettings

class Config(BaseSettings):
    # Global configuration settings
    retrieval_method: str = "default"
    evaluation_categories: list[str] = ["relevant", "irrelevant"]
    scoring_rubric: dict = {"relevant": 1, "irrelevant": 0}

    class Config:
        env_prefix = "SAFEVAL_"
