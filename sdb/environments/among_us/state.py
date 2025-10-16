"""Game state for Among Us."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

# from sdb.core.base_state import BaseState  # Not needed - using plain dataclass
from sdb.environments.among_us.types import (
    MeetingResult,
    Phase,
    PlayerRole,
    PlayerState,
    TaskRoundResult,
)

if TYPE_CHECKING:
    from sdb.environments.among_us.map import SpaceshipMap


@dataclass
class AmongUsState:
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
        ship_map: Reference to the spatial map (not serialized)
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
    discussion_round: int = 0  # Current discussion round number
    players_spoken_this_round: set = field(default_factory=set)  # Players who spoke this round
    
    # Game outcome
    winner: Optional[str] = None
    win_reason: str = ""
    
    # Spatial (not serialized, managed by environment)
    ship_map: Optional['SpaceshipMap'] = field(default=None, repr=False, compare=False)
    
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
    
    # Spatial methods
    def get_players_in_room(self, room_name: str, alive_only: bool = True) -> List[int]:
        """Get all players in a specific room.
        
        Args:
            room_name: Name of the room
            alive_only: Only return alive players
            
        Returns:
            List of player IDs in the room
        """
        players_in_room = []
        for pid, player in self.players.items():
            if player.location == room_name:
                if not alive_only or player.is_alive:
                    players_in_room.append(pid)
        return players_in_room
    
    def get_visible_players(self, observer_pid: int) -> List[int]:
        """Get players visible to an observer (same room, alive).
        
        Args:
            observer_pid: Observer player ID
            
        Returns:
            List of visible player IDs
        """
        observer = self.players.get(observer_pid)
        if not observer or not observer.is_alive:
            return []
        
        # Can see alive players in same room (except self)
        visible = []
        for pid, player in self.players.items():
            if pid != observer_pid and player.is_alive and player.location == observer.location:
                visible.append(pid)
        return visible
    
    def can_player_move_to(self, player_id: int, target_room: str) -> bool:
        """Check if a player can move to a target room.
        
        Args:
            player_id: ID of the player
            target_room: Target room name
            
        Returns:
            True if movement is legal
        """
        player = self.players.get(player_id)
        if not player or not player.is_alive or not self.ship_map:
            return False
        
        current_room = player.location
        
        # Can always move via corridor
        if self.ship_map.can_move_via_corridor(current_room, target_room):
            return True
        
        # Impostors can also move via vents
        if player.role == PlayerRole.IMPOSTOR:
            if self.ship_map.can_move_via_vent(current_room, target_room):
                return True
        
        return False

