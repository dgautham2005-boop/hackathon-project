import pandas as pd
from typing import List

def rank_by_value(predictions: pd.DataFrame, budget: int) -> List[str]:
    """Sorts directly by unconditional predicted value to maximize budget."""
    # Sort by value descending, then CustomerID descending to handle ties deterministically
    sorted_df = predictions.sort_values(by=['predicted_value', 'CustomerID'], ascending=[False, False])
    return sorted_df['CustomerID'].head(budget).tolist()

def evaluate_selection(selected_ids: List[str], actuals: pd.DataFrame, budget: int) -> float:
    """Calculates realized value capture score (max 30 points)."""
    if len(selected_ids) != budget or len(set(selected_ids)) != len(selected_ids):
        return 0.0 # Invalid selection size or duplicates
        
    selected_actuals = actuals[actuals['CustomerID'].isin(selected_ids)]
    realized_value = selected_actuals['actual_value'].sum()
    
    oracle_best = actuals.sort_values(by='actual_value', ascending=False).head(budget)
    oracle_value = oracle_best['actual_value'].sum()
    
    if oracle_value == 0:
        return 30.0
        
    return 30.0 * (realized_value / oracle_value)