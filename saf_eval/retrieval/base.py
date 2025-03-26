from abc import ABC, abstractmethod
from typing import List

from ..core.models import AtomicFact, RetrievedDocument

class RetrieverBase(ABC):
    """Base class for document retrieval implementations."""
    
    @abstractmethod
    async def retrieve(self, fact: AtomicFact, **kwargs) -> List[RetrievedDocument]:
        """Retrieve documents relevant to the given atomic fact."""
        pass
