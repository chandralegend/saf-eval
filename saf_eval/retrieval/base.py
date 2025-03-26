from abc import ABC, abstractmethod
from typing import List

from ..config import Config
from ..core.models import AtomicFact, RetrievedDocument

class RetrieverBase(ABC):
    """Base class for document retrieval implementations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    @abstractmethod
    async def retrieve(self, fact: AtomicFact, **kwargs) -> List[RetrievedDocument]:
        """Retrieve documents relevant to the given atomic fact."""
        pass
