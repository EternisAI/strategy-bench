"""Visualization tools for game analysis."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json


class Visualizer:
    """Creates visualizations from game data.
    
    Note: Actual plotting would require matplotlib/seaborn.
    This provides the data preparation and structure.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else Path("visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_win_rates(self, metrics: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for win rate visualization.
        
        Args:
            metrics: Player metrics
            
        Returns:
            Plot data dictionary
        """
        plot_data = {
            "type": "bar",
            "title": "Win Rates by Player",
            "x_label": "Player ID",
            "y_label": "Win Rate",
            "data": []
        }
        
        for player_id, player_metrics in metrics.items():
            win_rate = player_metrics["game_metrics"]["win_rate"]
            plot_data["data"].append({
                "x": f"Player {player_id}",
                "y": win_rate
            })
        
        return plot_data
    
    def plot_deception_success(self, metrics: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for deception success visualization.
        
        Args:
            metrics: Player metrics
            
        Returns:
            Plot data dictionary
        """
        plot_data = {
            "type": "scatter",
            "title": "Deception Success vs Accusation Accuracy",
            "x_label": "Deception Success Rate",
            "y_label": "Accusation Accuracy",
            "data": []
        }
        
        for player_id, player_metrics in metrics.items():
            deception = player_metrics["deception_metrics"]
            plot_data["data"].append({
                "x": deception["deception_success_rate"],
                "y": deception["accusation_accuracy"],
                "label": f"Player {player_id}"
            })
        
        return plot_data
    
    def plot_game_length_distribution(
        self,
        tournament_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for game length distribution.
        
        Args:
            tournament_data: Tournament result data
            
        Returns:
            Plot data dictionary
        """
        game_lengths = [game["num_rounds"] for game in tournament_data["games"]]
        
        plot_data = {
            "type": "histogram",
            "title": "Game Length Distribution",
            "x_label": "Number of Rounds",
            "y_label": "Frequency",
            "data": game_lengths
        }
        
        return plot_data
    
    def create_dashboard(
        self,
        tournament_path: Path,
        metrics_path: Path
    ) -> Dict[str, Any]:
        """Create dashboard data with multiple visualizations.
        
        Args:
            tournament_path: Path to tournament results
            metrics_path: Path to metrics summary
            
        Returns:
            Dashboard data dictionary
        """
        # Load data
        with open(tournament_path, 'r') as f:
            tournament_data = json.load(f)
        
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        
        # Create visualizations
        dashboard = {
            "title": f"Tournament Analysis: {tournament_data['tournament_id']}",
            "summary": {
                "num_games": len(tournament_data["games"]),
                "num_players": len(metrics_data["players"]),
                "duration": tournament_data.get("duration_seconds", 0),
            },
            "plots": {
                "win_rates": self.plot_win_rates(metrics_data["players"]),
                "deception": self.plot_deception_success(metrics_data["players"]),
                "game_lengths": self.plot_game_length_distribution(tournament_data),
            }
        }
        
        # Save dashboard data
        dashboard_file = self.output_dir / "dashboard.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        return dashboard
    
    def export_for_plotting(
        self,
        plot_data: Dict[str, Any],
        filename: str
    ) -> Path:
        """Export plot data to JSON for external plotting.
        
        Args:
            plot_data: Plot data dictionary
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(plot_data, f, indent=2)
        return output_path

