"""Avalon game state."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import random

from .config import AvalonConfig
from .types import (
    Role, Team, Phase, PlayerState, QuestResult, TeamProposal, DiscussionStatement,
    QUEST_SIZES, TEAM_COMPOSITION, QUEST_FAILS_NEEDED,
)


@dataclass
class AvalonState:
    """Complete game state for Avalon.
    
    Attributes:
        config: Game configuration
        rng: Random number generator
        players: List of player states
        quest_leader: Index of current quest leader
        current_phase: Current game phase
        current_quest: Current quest number (0-4)
        current_round: Current round number (within a quest, for team proposals)
        quest_results: List of completed quest results
        current_proposal: Current team proposal
        proposal_history: History of all proposals this quest
        team_rejections: Number of consecutive team rejections
        quests_succeeded: Number of quests that succeeded
        quests_failed: Number of quests that failed
        game_over: Whether the game has ended
        winner: Winning team (or None)
    """

    config: AvalonConfig
    rng: random.Random
    
    # Players
    players: List[PlayerState] = field(default_factory=list)
    
    # Quest tracking
    quest_leader: int = 0
    current_phase: Phase = Phase.TEAM_SELECTION
    current_quest: int = 0  # 0-4 for 5 quests
    current_round: int = 0  # Proposal attempts within a quest
    
    # Quest results
    quest_results: List[QuestResult] = field(default_factory=list)
    
    # Current team proposal
    current_proposal: Optional[TeamProposal] = None
    proposal_history: List[TeamProposal] = field(default_factory=list)
    team_rejections: int = 0
    total_proposals: int = 0  # Global proposal counter (increments with each proposal)
    
    # Discussion tracking
    current_discussion: List[DiscussionStatement] = field(default_factory=list)
    discussion_order: List[int] = field(default_factory=list)  # Order players speak in
    next_speaker_index: int = 0  # Index in discussion_order for next speaker
    
    # Vote tracking (prevent double voting and track votes)
    team_votes_cast: set = field(default_factory=set)  # Players who voted in current team voting
    quest_votes_by_player: Dict[int, str] = field(default_factory=dict)  # player_id -> "success"/"fail"
    quest_voters_done: set = field(default_factory=set)  # Players who completed quest voting
    
    # Discussion tracking (prevent repeated speaking)
    spoken_this_round: set = field(default_factory=set)  # Players who spoke in current discussion round
    
    # Score tracking
    quests_succeeded: int = 0
    quests_failed: int = 0
    
    # Game end
    game_over: bool = False
    winner: Optional[Team] = None
    assassin_target: Optional[int] = None
    
    def get_player(self, pid: int) -> PlayerState:
        """Get player state by ID."""
        return self.players[pid]
    
    def get_team_size(self) -> int:
        """Get required team size for current quest."""
        return QUEST_SIZES[self.config.n_players][self.current_quest]
    
    def get_fails_needed(self) -> int:
        """Get number of fails needed for current quest to fail."""
        return QUEST_FAILS_NEEDED[self.config.n_players][self.current_quest]
    
    def advance_quest_leader(self):
        """Move quest leader to next player."""
        self.quest_leader = (self.quest_leader + 1) % self.config.n_players
    
    def get_role_visibility(self, pid: int) -> Dict[int, Optional[Team]]:
        """Get what teams the player can see.
        
        Returns dict mapping player_id -> Team (or None if unknown)
        """
        player = self.get_player(pid)
        visibility = {}
        
        for other_pid, other_player in enumerate(self.players):
            if other_pid == pid:
                # Player knows their own team
                visibility[other_pid] = other_player.team
            elif player.role == Role.MERLIN:
                # Merlin sees all evil except Mordred
                if other_player.team == Team.EVIL and other_player.role != Role.MORDRED:
                    visibility[other_pid] = Team.EVIL
                else:
                    visibility[other_pid] = None
            elif player.role == Role.PERCIVAL:
                # Percival sees Merlin and Morgana (but doesn't know which is which)
                if other_player.role in (Role.MERLIN, Role.MORGANA):
                    visibility[other_pid] = Team.GOOD  # Sees them as potentially Merlin
                else:
                    visibility[other_pid] = None
            elif player.team == Team.EVIL and player.role != Role.OBERON:
                # Evil players (except Oberon) see each other
                if other_player.team == Team.EVIL and other_player.role != Role.OBERON:
                    visibility[other_pid] = Team.EVIL
                else:
                    visibility[other_pid] = None
            else:
                # Regular servants and Oberon don't see anything
                visibility[other_pid] = None
        
        return visibility
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "config": {
                "n_players": self.config.n_players,
                "seed": self.config.seed,
            },
            "quest_leader": self.quest_leader,
            "current_phase": self.current_phase.value,
            "current_quest": self.current_quest,
            "current_round": self.current_round,
            "team_rejections": self.team_rejections,
            "quests_succeeded": self.quests_succeeded,
            "quests_failed": self.quests_failed,
            "game_over": self.game_over,
            "winner": self.winner.value if self.winner else None,
            "players": [
                {
                    "pid": p.pid,
                    "role": p.role.value,
                    "team": p.team.value,
                    "is_alive": p.is_alive,
                }
                for p in self.players
            ],
        }

