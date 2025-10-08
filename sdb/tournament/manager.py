"""Tournament manager for running multiple tournaments."""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from sdb.core.base_agent import BaseAgent
from sdb.tournament.base import BaseTournament, TournamentConfig, TournamentResult
from sdb.tournament.round_robin import RoundRobinTournament
from sdb.tournament.swiss import SwissTournament


class TournamentManager:
    """Manages multiple tournaments and tracks overall statistics."""
    
    def __init__(self, output_dir: Path):
        """Initialize tournament manager.
        
        Args:
            output_dir: Directory to save tournament results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[TournamentResult] = []
    
    async def run_tournament(
        self,
        tournament_type: str,
        config: TournamentConfig,
        agents: List[BaseAgent]
    ) -> TournamentResult:
        """Run a tournament.
        
        Args:
            tournament_type: Type of tournament ("round_robin", "swiss")
            config: Tournament configuration
            agents: List of agents to compete
            
        Returns:
            Tournament results
        """
        # Create tournament
        if tournament_type == "round_robin":
            tournament = RoundRobinTournament(config)
        elif tournament_type == "swiss":
            tournament = SwissTournament(config)
        else:
            raise ValueError(f"Unknown tournament type: {tournament_type}")
        
        # Run tournament
        result = await tournament.run(agents)
        self.results.append(result)
        
        return result
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get statistics across all tournaments.
        
        Returns:
            Dictionary of overall statistics
        """
        if not self.results:
            return {}
        
        total_games = sum(len(r.games) for r in self.results)
        total_duration = sum(
            (r.end_time - r.start_time).total_seconds()
            for r in self.results
        )
        
        return {
            "num_tournaments": len(self.results),
            "total_games": total_games,
            "total_duration_seconds": total_duration,
            "average_game_duration": total_duration / total_games if total_games > 0 else 0,
        }
    
    def save_all_results(self) -> None:
        """Save all tournament results to disk."""
        for result in self.results:
            filepath = self.output_dir / f"{result.tournament_id}.json"
            result.save(filepath)

