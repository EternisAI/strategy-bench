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


# Voting system constants and utilities
SKIP_TOKEN = "skip"  # Keep for backward compatibility in observations

def normalize_target(t):
    """Accept "Agent_5", "Player 5", "5", 5, "skip" and normalize to int or None.
    
    Returns:
        int | None: Player ID or None for skip
    """
    if t is None:
        return None
    if isinstance(t, int):
        return t
    if isinstance(t, str):
        ts = t.strip().lower()
        if ts in {"skip", "none", ""}:
            return None
        # Extract digits from "Agent_5", "Player 5", "5"
        digits = "".join(ch for ch in ts if ch.isdigit())
        if digits.isdigit():
            return int(digits)
        return None
    return None

def legal_vote_targets(alive_ids: set[int], voter_id: int) -> List[int | None]:
    """Get legal vote targets for a voter (now includes self-vote per AU rules)."""
    return sorted([i for i in alive_ids]) + [None]  # None = skip

def cast_vote(meeting: 'MeetingState', voter_id: int, raw_target):
    """Cast a vote with validation.
    
    Args:
        meeting: Current meeting state
        voter_id: ID of voting player
        raw_target: Raw target (can be string, int, or None)
        
    Returns:
        dict with "ok": True or "error": str
    """
    tgt = normalize_target(raw_target)
    
    # Check voter is alive
    if voter_id not in meeting.alive:
        return {"error": "NOT_ELIGIBLE"}
    
    # Check target is valid (alive or None for skip)
    # Note: Self-voting is now allowed per Among Us rules
    if tgt is not None and tgt not in meeting.alive:
        return {"error": "INVALID_TARGET", "allowed": legal_vote_targets(meeting.alive, voter_id)}
    
    # Replacement semantics: last vote wins
    meeting.ballots[voter_id] = tgt
    return {"ok": True}

from collections import Counter

def close_meeting(meeting: 'MeetingState'):
    """Close a meeting and determine the result.
    
    Returns:
        dict with:
            - "ejected": int (player ID) if someone is ejected
            - "skipped": bool (True if no ejection)
            - "votes": dict of all ballots
    """
    # Invariant: number of ballots <= number of alive voters
    tally = Counter(meeting.ballots.values())
    
    # If no votes cast, skip
    if not tally:
        return {"skipped": True, "votes": {}}
    
    # Remove None (skip votes) from winner contention
    non_skip = {k: v for k, v in tally.items() if k is not None}
    
    # If only skip votes, skip
    if not non_skip:
        return {"skipped": True, "votes": dict(meeting.ballots)}
    
    # Find player(s) with most votes
    max_votes = max(non_skip.values())
    top = [k for k, v in non_skip.items() if v == max_votes]
    
    # If tie among multiple players, skip (no ejection)
    if len(top) > 1:
        return {"skipped": True, "votes": dict(meeting.ballots)}
    
    # Single plurality winner -> eject
    ejected = top[0]
    return {"ejected": ejected, "skipped": False, "votes": dict(meeting.ballots)}


@dataclass
class MeetingState:
    """State for a single meeting/voting session."""
    def __init__(self, alive_ids: list[int]):
        self.alive = set(alive_ids)
        self.ballots: dict[int, int | None] = {}  # voter -> target_id or None (skip)


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
    meeting: Optional[MeetingState] = field(default=None)  # Current meeting state
    
    # Timeout tracking for phases
    voting_started_round: int = 0  # Round number when voting started
    discussion_started_round: int = 0  # Round number when discussion started
    discussion_attempts: int = 0  # Number of attempts in current discussion round
    
    # Meeting history for formatted display
    meeting_history: List[dict] = field(default_factory=list)  # List of meeting summaries
    
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
        """Decrease all impostor kill cooldowns by 1 and clean up dead impostors."""
        alive_impostors = set(self.get_alive_impostors())
        
        for pid in list(self.impostor_kill_cooldowns.keys()):
            # Clean up dead impostors
            if pid not in alive_impostors:
                del self.impostor_kill_cooldowns[pid]
            # Decrease cooldown for alive impostors
            elif self.impostor_kill_cooldowns[pid] > 0:
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

