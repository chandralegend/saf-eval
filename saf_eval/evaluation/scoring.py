from typing import Dict, List

from ..core.models import FactEvaluation, ResponseEvaluation

class FactualityScorer:
    """Scores the factuality of a response based on fact evaluations."""
    
    def __init__(self, scoring_rubric: Dict[str, float] = None):
        self.scoring_rubric = scoring_rubric or {"relevant": 1.0, "irrelevant": 0.0}
    
    def score(self, response_text: str, context: str, evaluations: List[FactEvaluation]) -> ResponseEvaluation:
        """Calculate the overall factuality score."""
        if not evaluations:
            return ResponseEvaluation(
                response_text=response_text,
                context=context,
                facts=[e.fact for e in evaluations],
                evaluations=evaluations,
                factuality_score=0.0
            )
        
        # Calculate weighted score based on the rubric
        total_score = 0.0
        for eval in evaluations:
            category_score = self.scoring_rubric.get(eval.category, 0.0)
            total_score += category_score * eval.confidence
        
        # Normalize score
        factuality_score = total_score / len(evaluations)
        
        return ResponseEvaluation(
            response_text=response_text,
            context=context,
            facts=[e.fact for e in evaluations],
            evaluations=evaluations,
            factuality_score=factuality_score
        )
