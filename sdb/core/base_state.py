"""Base state class for all game environments."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from sdb.core.types import GamePhase, Action, Observation, PlayerID
from sdb.core.exceptions import InvalidStateError


@dataclass
class BaseState(ABC):
    """Abstract base class for game state.
    
    All game-specific states should inherit from this class and implement
    the abstract methods.
    """
    
    # Core state attributes
    game_id: str
    num_players: int
    current_phase: GamePhase
    round_number: int = 0
    is_terminal: bool = False
    
    # Player tracking
    alive_players: List[PlayerID] = field(default_factory=list)
    player_roles: Dict[PlayerID, str] = field(default_factory=dict)
    
    # History
    action_history: List[Action] = field(default_factory=list)
    phase_history: List[GamePhase] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if not self.alive_players:
            self.alive_players = list(range(self.num_players))
        if not self.player_roles:
            self.player_roles = {i: "unknown" for i in range(self.num_players)}
    
    @abstractmethod
    def get_observation(self, player_id: PlayerID) -> Observation:
        """Get observation for a specific player.
        
        Args:
            player_id: ID of the player requesting observation
            
        Returns:
            Observation object containing what the player can see
        """
        pass
    
    @abstractmethod
    def is_action_legal(self, action: Action) -> bool:
        """Check if an action is legal in the current state.
        
        Args:
            action: The action to validate
            
        Returns:
            True if action is legal, False otherwise
        """
        pass
    
    @abstractmethod
    def get_legal_actions(self, player_id: PlayerID) -> List[Action]:
        """Get all legal actions for a player in the current state.
        
        Args:
            player_id: ID of the player
            
        Returns:
            List of legal actions
        """
        pass
    
    @abstractmethod
    def copy(self) -> "BaseState":
        """Create a deep copy of the state.
        
        Returns:
            Deep copy of this state
        """
        pass
    
    def add_action(self, action: Action) -> None:
        """Add action to history.
        
        Args:
            action: Action to add to history
        """
        self.action_history.append(action)
    
    def set_phase(self, phase: GamePhase) -> None:
        """Set current game phase.
        
        Args:
            phase: New game phase
        """
        self.phase_history.append(self.current_phase)
        self.current_phase = phase
    
    def is_player_alive(self, player_id: PlayerID) -> bool:
        """Check if a player is alive.
        
        Args:
            player_id: ID of the player
            
        Returns:
            True if player is alive, False otherwise
        """
        return player_id in self.alive_players
    
    def eliminate_player(self, player_id: PlayerID) -> None:
        """Eliminate a player from the game.
        
        Args:
            player_id: ID of the player to eliminate
        """
        if player_id in self.alive_players:
            self.alive_players.remove(player_id)
    
    def get_player_role(self, player_id: PlayerID) -> str:
        """Get the role of a player.
        
        Args:
            player_id: ID of the player
            
        Returns:
            Role name as string
        """
        return self.player_roles.get(player_id, "unknown")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation.
        
        Returns:
            Dictionary representation of the state
        """
        return {
            "game_id": self.game_id,
            "num_players": self.num_players,
            "current_phase": self.current_phase.name,
            "round_number": self.round_number,
            "is_terminal": self.is_terminal,
            "alive_players": self.alive_players,
            "player_roles": self.player_roles,
            "action_count": len(self.action_history),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def __str__(self) -> str:
        """String representation of the state."""
        return (
            f"{self.__class__.__name__}("
            f"game_id={self.game_id}, "
            f"phase={self.current_phase.name}, "
            f"round={self.round_number}, "
            f"alive={len(self.alive_players)}/{self.num_players})"
        )

