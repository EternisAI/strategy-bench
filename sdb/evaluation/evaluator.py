"""Evaluator for analyzing game logs and computing metrics."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from sdb.evaluation.metrics import AgentMetrics, GameMetrics, DeceptionMetrics, CommunicationMetrics
from sdb.logging.formats import LogEntry, EventType


class Evaluator:
    """Evaluates agent performance from game logs."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics: Dict[int, AgentMetrics] = {}
    
    def evaluate_game_log(self, log_path: Path) -> Dict[int, Dict[str, Any]]:
        """Evaluate a single game from log file.
        
        Args:
            log_path: Path to game log (JSONL format)
            
        Returns:
            Dictionary mapping player ID to metrics
        """
        # Load log entries
        entries = []
        with open(log_path, 'r') as f:
            for line in f:
                entry_dict = json.loads(line)
                entry = LogEntry.from_dict(entry_dict)
                entries.append(entry)
        
        # Extract game result
        game_result = self._extract_game_result(entries)
        
        # Update metrics for each player
        player_metrics = {}
        for player_id in self._get_player_ids(entries):
            if player_id not in self.metrics:
                self.metrics[player_id] = AgentMetrics(
                    agent_id=player_id,
                    agent_name=f"Player_{player_id}"
                )
            
            # Update game metrics
            won = self._check_if_won(player_id, game_result)
            game_length = game_result.get("num_rounds", 0)
            survival = self._calculate_survival(player_id, entries)
            
            self.metrics[player_id].game_metrics.update(won, game_length, survival)
            
            # Update deception and communication metrics from log
            self._update_from_log(player_id, entries)
            
            player_metrics[player_id] = self.metrics[player_id].to_dict()
        
        return player_metrics
    
    def evaluate_tournament(self, tournament_result_path: Path) -> Dict[int, Dict[str, Any]]:
        """Evaluate all games in a tournament.
        
        Args:
            tournament_result_path: Path to tournament result JSON
            
        Returns:
            Dictionary mapping player ID to aggregated metrics
        """
        with open(tournament_result_path, 'r') as f:
            tournament_data = json.load(f)
        
        # Process each game
        for game in tournament_data["games"]:
            # If log files are available, evaluate them
            log_path = game.get("log_filepath")
            if log_path and Path(log_path).exists():
                self.evaluate_game_log(Path(log_path))
        
        return {pid: metrics.to_dict() for pid, metrics in self.metrics.items()}
    
    def _extract_game_result(self, entries: List[LogEntry]) -> Dict[str, Any]:
        """Extract game result from log entries.
        
        Args:
            entries: Log entries
            
        Returns:
            Game result dictionary
        """
        for entry in reversed(entries):
            if entry.event_type == EventType.GAME_END:
                return entry.data
        return {}
    
    def _get_player_ids(self, entries: List[LogEntry]) -> List[int]:
        """Get all player IDs from log.
        
        Args:
            entries: Log entries
            
        Returns:
            List of player IDs
        """
        player_ids = set()
        for entry in entries:
            if entry.player_id is not None:
                player_ids.add(entry.player_id)
        return sorted(list(player_ids))
    
    def _check_if_won(self, player_id: int, game_result: Dict[str, Any]) -> bool:
        """Check if player won the game.
        
        Args:
            player_id: Player ID
            game_result: Game result dictionary
            
        Returns:
            True if player won
        """
        winner = game_result.get("winner")
        if isinstance(winner, list):
            return player_id in winner
        return str(player_id) == str(winner)
    
    def _calculate_survival(self, player_id: int, entries: List[LogEntry]) -> int:
        """Calculate how many rounds player survived.
        
        Args:
            player_id: Player ID
            entries: Log entries
            
        Returns:
            Number of rounds survived
        """
        max_round = 0
        eliminated_round = None
        
        for entry in entries:
            if entry.round_number > max_round:
                max_round = entry.round_number
            
            if (entry.event_type == EventType.PLAYER_ELIMINATED and
                entry.player_id == player_id):
                eliminated_round = entry.round_number
                break
        
        return eliminated_round if eliminated_round else max_round
    
    def _update_from_log(self, player_id: int, entries: List[LogEntry]) -> None:
        """Update metrics from log entries.
        
        Args:
            player_id: Player ID
            entries: Log entries
        """
        metrics = self.metrics[player_id]
        
        for entry in entries:
            if entry.player_id != player_id:
                continue
            
            # Update based on event type
            if entry.event_type == EventType.PLAYER_SPEAK:
                message = entry.data.get("message", "")
                metrics.communication_metrics.update_message(len(message))
            
            elif entry.event_type == EventType.PLAYER_VOTE:
                # Could track voting patterns here
                pass
            
            # Add more event type handling as needed
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluated metrics.
        
        Returns:
            Summary dictionary
        """
        return {
            "num_players": len(self.metrics),
            "players": {
                pid: metrics.to_dict()
                for pid, metrics in self.metrics.items()
            }
        }
    
    def save_summary(self, output_path: Path) -> None:
        """Save evaluation summary to file.
        
        Args:
            output_path: Path to save summary
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)

