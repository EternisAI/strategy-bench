"""Secret Hitler game state management."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from copy import deepcopy

from sdb.core.base_state import BaseState
from sdb.core.types import Action, Observation, ObservationType, PlayerID, GamePhase
from sdb.environments.secret_hitler.types import (
    PlayerInfo, Government, Policy, Party, Role, Phase, PresidentialPower
)
from sdb.environments.secret_hitler.rules import PolicyDeck


@dataclass
class SecretHitlerState(BaseState):
    """Game state for Secret Hitler.
    
    Maintains both public information (visible to all) and private
    information (known only to specific players).
    """
    
    # Public state
    phase: Phase = Phase.ELECTION_NOMINATION
    liberal_policies: int = 0
    fascist_policies: int = 0
    election_tracker: int = 0
    veto_unlocked: bool = False  # Unlocks after 5 Fascist policies
    
    president_idx: int = 0
    chancellor_nominee: Optional[int] = None
    last_government: Optional[Government] = None
    
    votes: List = field(default_factory=list)
    current_discussion: List[Dict] = field(default_factory=list)
    
    # Presidential power state
    pending_power: Optional[PresidentialPower] = None
    investigated_players: List[int] = field(default_factory=list)
    confirmed_not_hitler: set = field(default_factory=set)
    
    # Special election tracking
    special_election_pending: bool = False
    special_election_return_to: Optional[int] = None
    
    # Game history
    game_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Private state (only accessed by environment, not visible in observations)
    players: List[PlayerInfo] = field(default_factory=list)
    policy_deck: Optional[PolicyDeck] = None
    president_hand: List[Policy] = field(default_factory=list)
    chancellor_hand: List[Policy] = field(default_factory=list)
    
    def get_observation(self, player_id: PlayerID) -> Observation:
        """Get observation for a specific player.
        
        Args:
            player_id: ID of the player
            
        Returns:
            Observation with public + private info for this player
        """
        # Public information
        public_data = {
            "phase": self.phase.name,
            "liberal_policies": self.liberal_policies,
            "fascist_policies": self.fascist_policies,
            "election_tracker": self.election_tracker,
            "veto_unlocked": self.veto_unlocked,
            "president_idx": self.president_idx,
            "chancellor_nominee": self.chancellor_nominee,
            "last_government": self.last_government,
            "alive_players": self.alive_players,
            "confirmed_not_hitler": list(self.confirmed_not_hitler),
            "game_history": self.game_history[-5:],  # Last 5 events
            "current_discussion": self.current_discussion,
        }
        
        # Private information for this player
        player_info = self.players[player_id]
        private_data = {
            "your_role": player_info.role.name,
            "your_party": player_info.party.name,
        }
        
        # Fascists know each other and Hitler
        if player_info.is_fascist():
            fascist_ids = [p.player_id for p in self.players if p.is_fascist()]
            hitler_id = next((p.player_id for p in self.players if p.is_hitler), None)
            private_data["fascist_team"] = fascist_ids
            private_data["hitler_id"] = hitler_id
        
        # Add hands if in legislative session
        if self.phase == Phase.LEGISLATIVE_SESSION:
            if player_id == self.president_idx and self.president_hand:
                private_data["your_hand"] = [p.name for p in self.president_hand]
            elif player_id == self.last_government.chancellor and self.chancellor_hand:
                private_data["your_hand"] = [p.name for p in self.chancellor_hand]
        
        return Observation(
            player_id=player_id,
            obs_type=ObservationType.PRIVATE if private_data else ObservationType.PUBLIC,
            phase=GamePhase.VOTING if self.phase == Phase.ELECTION_VOTING else GamePhase.DISCUSSION,
            data={**public_data, **private_data}
        )
    
    def is_action_legal(self, action: Action) -> bool:
        """Check if an action is legal in current state.
        
        Args:
            action: Action to validate
            
        Returns:
            True if legal
        """
        # Basic validation
        if not self.is_player_alive(action.player_id):
            return False
        
        # Phase-specific validation
        if self.phase == Phase.ELECTION_NOMINATION:
            return action.player_id == self.president_idx
        elif self.phase == Phase.ELECTION_VOTING:
            return action.player_id in self.alive_players
        elif self.phase == Phase.LEGISLATIVE_SESSION:
            return (action.player_id == self.president_idx or 
                    action.player_id == self.last_government.chancellor)
        
        return True
    
    def get_legal_actions(self, player_id: PlayerID) -> List[Action]:
        """Get legal actions for a player.
        
        Args:
            player_id: Player ID
            
        Returns:
            List of legal actions (simplified for base implementation)
        """
        # This would be expanded in the full implementation
        return []
    
    def copy(self) -> "SecretHitlerState":
        """Create deep copy of state."""
        return deepcopy(self)
    
    def living_count(self) -> int:
        """Get number of living players."""
        return len(self.alive_players)
    
    def get_player(self, player_id: int) -> PlayerInfo:
        """Get player info."""
        return self.players[player_id]

