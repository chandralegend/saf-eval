import pytest
from saf_eval.config import Config

# Fixture for shared configuration object
@pytest.fixture
def config():
    return Config()

# Note: pytest-asyncio will be automatically used because we configured
# asyncio_mode = "auto" in pyproject.toml
