"""Base tournament class."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from sdb.core.base_agent import BaseAgent
from sdb.core.base_env import BaseEnvironment


@dataclass
class TournamentConfig:
    """Configuration for a tournament."""
    
    name: str
    environment: str
    num_games: int
    output_dir: Path
    log_games: bool = True
    log_private_info: bool = False
    parallel_games: int = 1
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameRecord:
    """Record of a single game in a tournament."""
    
    game_id: str
    game_number: int
    players: List[int]  # Player IDs
    winner: Any
    win_reason: str
    num_rounds: int
    duration_seconds: float
    player_stats: Dict[int, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "game_number": self.game_number,
            "players": self.players,
            "winner": str(self.winner),
            "win_reason": self.win_reason,
            "num_rounds": self.num_rounds,
            "duration_seconds": self.duration_seconds,
            "player_stats": self.player_stats,
            "metadata": self.metadata,
        }


@dataclass
class TournamentResult:
    """Results of a completed tournament."""
    
    tournament_id: str
    config: TournamentConfig
    start_time: datetime
    end_time: datetime
    games: List[GameRecord]
    player_stats: Dict[int, Dict[str, Any]]
    rankings: List[tuple]  # (player_id, score)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tournament_id": self.tournament_id,
            "config": {
                "name": self.config.name,
                "environment": self.config.environment,
                "num_games": self.config.num_games,
                "metadata": self.config.metadata,
            },
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "num_games": len(self.games),
            "games": [g.to_dict() for g in self.games],
            "player_stats": self.player_stats,
            "rankings": [(str(pid), score) for pid, score in self.rankings],
            "metadata": self.metadata,
        }
    
    def save(self, filepath: Path) -> None:
        """Save results to JSON file.
        
        Args:
            filepath: Path to save to
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> "TournamentResult":
        """Load results from JSON file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            TournamentResult instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct objects
        config = TournamentConfig(
            name=data["config"]["name"],
            environment=data["config"]["environment"],
            num_games=data["config"]["num_games"],
            output_dir=Path(""),
            metadata=data["config"]["metadata"]
        )
        
        games = [
            GameRecord(
                game_id=g["game_id"],
                game_number=g["game_number"],
                players=g["players"],
                winner=g["winner"],
                win_reason=g["win_reason"],
                num_rounds=g["num_rounds"],
                duration_seconds=g["duration_seconds"],
                player_stats=g["player_stats"],
                metadata=g["metadata"]
            )
            for g in data["games"]
        ]
        
        return cls(
            tournament_id=data["tournament_id"],
            config=config,
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            games=games,
            player_stats=data["player_stats"],
            rankings=[(int(pid), score) for pid, score in data["rankings"]],
            metadata=data["metadata"]
        )


class BaseTournament(ABC):
    """Abstract base class for tournaments."""
    
    def __init__(self, config: TournamentConfig):
        """Initialize tournament.
        
        Args:
            config: Tournament configuration
        """
        self.config = config
        self.games: List[GameRecord] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    @abstractmethod
    async def run(self, agents: List[BaseAgent]) -> TournamentResult:
        """Run the tournament.
        
        Args:
            agents: List of agents to compete
            
        Returns:
            Tournament results
        """
        pass
    
    @abstractmethod
    def _generate_matchups(self, agents: List[BaseAgent]) -> List[List[int]]:
        """Generate matchups for all games.
        
        Args:
            agents: List of agents
            
        Returns:
            List of matchups (each matchup is a list of agent IDs)
        """
        pass
    
    def _calculate_rankings(self, agents: List[BaseAgent]) -> List[tuple]:
        """Calculate final rankings based on game results.
        
        Args:
            agents: List of agents
            
        Returns:
            List of (agent_id, score) tuples sorted by score
        """
        scores = {agent.player_id: 0 for agent in agents}
        
        for game in self.games:
            # Award points based on winner
            # This is game-specific and should be customized
            if isinstance(game.winner, list):
                # Team win
                for player_id in game.winner:
                    scores[player_id] += 1
            else:
                # Individual or team name
                # Need to map back to player IDs
                pass
        
        # Sort by score descending
        rankings = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return rankings
    
    def _aggregate_player_stats(self, agents: List[BaseAgent]) -> Dict[int, Dict[str, Any]]:
        """Aggregate statistics for each player across all games.
        
        Args:
            agents: List of agents
            
        Returns:
            Dictionary mapping player ID to aggregated stats
        """
        stats = {agent.player_id: {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,
        } for agent in agents}
        
        for game in self.games:
            for player_id in game.players:
                stats[player_id]["games_played"] += 1
                stats[player_id]["total_rounds"] += game.num_rounds
                
                # Check if won
                if isinstance(game.winner, list) and player_id in game.winner:
                    stats[player_id]["wins"] += 1
                elif str(player_id) == str(game.winner):
                    stats[player_id]["wins"] += 1
                else:
                    stats[player_id]["losses"] += 1
        
        # Calculate win rate
        for player_id in stats:
            games = stats[player_id]["games_played"]
            if games > 0:
                stats[player_id]["win_rate"] = stats[player_id]["wins"] / games
            else:
                stats[player_id]["win_rate"] = 0.0
        
        return stats

