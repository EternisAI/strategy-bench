"""Statistical analysis tools."""

from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats as scipy_stats


class StatisticsAnalyzer:
    """Performs statistical analysis on game data."""
    
    def __init__(self):
        """Initialize statistics analyzer."""
        pass
    
    def compare_agents(
        self,
        agent1_wins: List[bool],
        agent2_wins: List[bool],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Compare performance of two agents statistically.
        
        Args:
            agent1_wins: List of win/loss for agent 1
            agent2_wins: List of win/loss for agent 2
            confidence_level: Confidence level for significance test
            
        Returns:
            Comparison results
        """
        # Calculate win rates
        wr1 = np.mean(agent1_wins)
        wr2 = np.mean(agent2_wins)
        
        # Perform proportion test
        if len(agent1_wins) > 0 and len(agent2_wins) > 0:
            stat, pvalue = scipy_stats.ttest_ind(
                agent1_wins,
                agent2_wins,
                equal_var=False
            )
            
            significant = pvalue < (1 - confidence_level)
        else:
            stat, pvalue, significant = 0.0, 1.0, False
        
        return {
            "agent1_win_rate": wr1,
            "agent2_win_rate": wr2,
            "difference": wr1 - wr2,
            "statistic": float(stat),
            "p_value": float(pvalue),
            "significant": significant,
            "confidence_level": confidence_level,
        }
    
    def calculate_confidence_interval(
        self,
        data: List[float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for data.
        
        Args:
            data: List of data points
            confidence_level: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(data) < 2:
            return (0.0, 0.0)
        
        mean = np.mean(data)
        std_err = scipy_stats.sem(data)
        margin = std_err * scipy_stats.t.ppf((1 + confidence_level) / 2, len(data) - 1)
        
        return (mean - margin, mean + margin)
    
    def calculate_effect_size(
        self,
        group1: List[float],
        group2: List[float]
    ) -> Dict[str, float]:
        """Calculate Cohen's d effect size.
        
        Args:
            group1: First group of measurements
            group2: Second group of measurements
            
        Returns:
            Effect size metrics
        """
        if len(group1) < 2 or len(group2) < 2:
            return {"cohens_d": 0.0, "interpretation": "insufficient_data"}
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            "cohens_d": cohens_d,
            "interpretation": interpretation,
            "pooled_std": pooled_std,
        }
    
    def analyze_variance(
        self,
        groups: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Perform ANOVA to compare multiple groups.
        
        Args:
            groups: Dictionary mapping group names to measurements
            
        Returns:
            ANOVA results
        """
        if len(groups) < 2:
            return {"error": "Need at least 2 groups for ANOVA"}
        
        group_data = list(groups.values())
        f_stat, p_value = scipy_stats.f_oneway(*group_data)
        
        return {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "num_groups": len(groups),
            "group_means": {
                name: float(np.mean(data))
                for name, data in groups.items()
            }
        }

