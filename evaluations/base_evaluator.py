"""Base evaluator class for analyzing game tournament logs."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class GameResult:
    """Standardized game result."""
    game_id: str
    winner: str
    win_reason: str
    num_rounds: int
    duration: float
    players: List[str]
    player_roles: Dict[str, str]
    player_stats: Dict[str, Dict[str, Any]]


class BaseEvaluator:
    """Base class for evaluating game tournaments."""
    
    def __init__(self, tournament_dir: Path):
        """Initialize evaluator.
        
        Args:
            tournament_dir: Path to tournament directory with logs/
        """
        self.tournament_dir = Path(tournament_dir)
        self.logs_dir = self.tournament_dir / "logs"
        self.results: List[GameResult] = []
        
    def load_all_games(self) -> List[GameResult]:
        """Load all game logs from tournament directory.
        
        Returns:
            List of game results
        """
        if not self.logs_dir.exists():
            raise ValueError(f"Logs directory not found: {self.logs_dir}")
        
        log_files = sorted(self.logs_dir.glob("*.jsonl"))
        print(f"Found {len(log_files)} game logs")
        
        for log_file in log_files:
            try:
                result = self.parse_game_log(log_file)
                if result:
                    self.results.append(result)
            except Exception as e:
                print(f"⚠️  Error parsing {log_file.name}: {e}")
        
        print(f"✓ Loaded {len(self.results)} games successfully")
        return self.results
    
    def parse_game_log(self, log_file: Path) -> Optional[GameResult]:
        """Parse a single game log file.
        
        Args:
            log_file: Path to .jsonl log file
            
        Returns:
            GameResult or None if parsing fails
        """
        raise NotImplementedError("Subclasses must implement parse_game_log")
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary statistics table.
        
        Returns:
            DataFrame with per-model statistics
        """
        if not self.results:
            raise ValueError("No results loaded. Call load_all_games() first.")
        
        # Collect stats per model
        model_stats = defaultdict(lambda: {
            'games_played': 0,
            'wins': 0,
            'total_rounds': 0,
        })
        
        for result in self.results:
            for player, role in result.player_roles.items():
                model_stats[player]['games_played'] += 1
                model_stats[player]['total_rounds'] += result.num_rounds
                
                # Check if winner
                if self._is_winner(player, result):
                    model_stats[player]['wins'] += 1
        
        # Calculate derived metrics
        rows = []
        for model, stats in sorted(model_stats.items()):
            win_rate = stats['wins'] / stats['games_played'] if stats['games_played'] > 0 else 0
            avg_rounds = stats['total_rounds'] / stats['games_played'] if stats['games_played'] > 0 else 0
            
            rows.append({
                'Model': model,
                'Games': stats['games_played'],
                'Wins': stats['wins'],
                'Win Rate': f"{win_rate:.1%}",
                'Avg Rounds': f"{avg_rounds:.1f}",
            })
        
        df = pd.DataFrame(rows)
        return df
    
    def _is_winner(self, player: str, result: GameResult) -> bool:
        """Check if player won the game.
        
        Args:
            player: Player/model name
            result: Game result
            
        Returns:
            True if player won
        """
        # Default implementation - override for team-based games
        return player in result.winner or result.winner == player
    
    def generate_detailed_stats(self) -> Dict[str, Any]:
        """Generate detailed statistics.
        
        Returns:
            Dictionary with detailed stats
        """
        raise NotImplementedError("Subclasses should implement detailed stats")
    
    def save_tables(self, output_dir: Optional[Path] = None):
        """Save all tables to files.
        
        Args:
            output_dir: Directory to save tables (default: tournament_dir)
        """
        if output_dir is None:
            output_dir = self.tournament_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary table
        summary_df = self.generate_summary_table()
        summary_file = output_dir / "summary_table.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"✓ Saved summary table: {summary_file}")
        
        # Detailed stats
        try:
            detailed_stats = self.generate_detailed_stats()
            stats_file = output_dir / "detailed_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(detailed_stats, f, indent=2)
            print(f"✓ Saved detailed stats: {stats_file}")
        except NotImplementedError:
            pass
        
        return summary_df
    
    def print_summary(self):
        """Print summary to console."""
        df = self.generate_summary_table()
        print("\n" + "="*80)
        print("TOURNAMENT SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")


def read_jsonl(file_path: Path) -> List[Dict]:
    """Read JSONL file.
    
    Args:
        file_path: Path to .jsonl file
        
    Returns:
        List of parsed JSON objects
    """
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def extract_model_name(full_name: str) -> str:
    """Extract clean model name from full agent name.
    
    Args:
        full_name: Full model name (e.g., "openai/gpt-4o")
        
    Returns:
        Clean name (e.g., "gpt-4o")
    """
    if '/' in full_name:
        return full_name.split('/')[-1]
    return full_name

