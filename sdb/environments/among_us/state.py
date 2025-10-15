"""Game state for Among Us."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sdb.core.base_state import BaseState
from sdb.environments.among_us.types import (
    MeetingResult,
    Phase,
    PlayerRole,
    PlayerState,
    TaskRoundResult,
)


@dataclass
class AmongUsState(BaseState):
    """Complete state of an Among Us game.
    
    Attributes:
        players: Dictionary of player states
        phase: Current game phase
        round_number: Current round number
        task_round_results: History of task rounds
        meeting_results: History of meetings
        impostor_kill_cooldowns: Rounds until each impostor can kill again
        current_meeting_statements: Statements in current meeting
        current_votes: Votes in current voting phase
        winner: Winning team ("crewmates" or "impostors")
        win_reason: Reason for game end
    """
    players: Dict[int, PlayerState] = field(default_factory=dict)
    phase: Phase = Phase.TASK
    round_number: int = 0
    
    # History
    task_round_results: List[TaskRoundResult] = field(default_factory=list)
    meeting_results: List[MeetingResult] = field(default_factory=list)
    
    # Impostor tracking
    impostor_kill_cooldowns: Dict[int, int] = field(default_factory=dict)
    
    # Current meeting
    current_meeting_statements: List[tuple[int, str]] = field(default_factory=list)
    current_votes: Dict[int, int] = field(default_factory=dict)
    
    # Game outcome
    winner: Optional[str] = None
    win_reason: str = ""
    
    def get_alive_players(self) -> List[int]:
        """Return list of alive player IDs."""
        return [pid for pid, p in self.players.items() if p.is_alive]
    
    def get_alive_crewmates(self) -> List[int]:
        """Return list of alive crewmate IDs."""
        return [
            pid for pid, p in self.players.items()
            if p.is_alive and p.role == PlayerRole.CREWMATE
        ]
    
    def get_alive_impostors(self) -> List[int]:
        """Return list of alive impostor IDs."""
        return [
            pid for pid, p in self.players.items()
            if p.is_alive and p.role == PlayerRole.IMPOSTOR
        ]
    
    def get_total_task_completion(self) -> float:
        """Get overall task completion progress (0.0 to 1.0)."""
        crewmates = [p for p in self.players.values() if p.role == PlayerRole.CREWMATE]
        if not crewmates:
            return 1.0
        
        total_completed = sum(p.tasks_completed for p in crewmates)
        total_tasks = sum(p.total_tasks for p in crewmates)
        
        return total_completed / total_tasks if total_tasks > 0 else 0.0
    
    def can_impostor_kill(self, impostor_id: int) -> bool:
        """Check if an impostor can kill (cooldown expired)."""
        return self.impostor_kill_cooldowns.get(impostor_id, 0) <= 0
    
    def decrease_kill_cooldowns(self):
        """Decrease all impostor kill cooldowns by 1."""
        for pid in list(self.impostor_kill_cooldowns.keys()):
            if self.impostor_kill_cooldowns[pid] > 0:
                self.impostor_kill_cooldowns[pid] -= 1

