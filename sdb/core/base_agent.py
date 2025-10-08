"""Base agent class for all agent implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from sdb.core.types import Action, Observation, PlayerID
from sdb.core.exceptions import AgentError


class BaseAgent(ABC):
    """Abstract base class for all agents.
    
    All agent implementations (LLM, search-based, baseline, human) should
    inherit from this class and implement the abstract methods.
    """
    
    def __init__(
        self,
        player_id: PlayerID,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize agent.
        
        Args:
            player_id: Unique identifier for this agent
            name: Human-readable name for the agent
            config: Configuration dictionary for the agent
        """
        self.player_id = player_id
        self.name = name or f"Agent_{player_id}"
        self.config = config or {}
        
        # Agent state
        self.observation_history: List[Observation] = []
        self.action_history: List[Action] = []
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Choose an action based on the observation.
        
        Args:
            observation: Current observation of the game state
            
        Returns:
            Action chosen by the agent
            
        Raises:
            AgentError: If agent fails to produce a valid action
        """
        pass
    
    def observe(self, observation: Observation) -> None:
        """Receive and store an observation.
        
        Args:
            observation: Observation to store
        """
        self.observation_history.append(observation)
    
    def record_action(self, action: Action) -> None:
        """Record an action taken by this agent.
        
        Args:
            action: Action to record
        """
        self.action_history.append(action)
    
    def reset(self) -> None:
        """Reset agent state for a new game.
        
        This should clear history and any internal state, but keep
        configuration and learned parameters.
        """
        self.observation_history.clear()
        self.action_history.clear()
        self.metadata.clear()
    
    def notify(self, event_type: str, data: Dict[str, Any]) -> None:
        """Receive notification about game events.
        
        Optional method for agents to process game events like:
        - Player elimination
        - Phase changes
        - Investigation results
        - etc.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this agent's behavior.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "player_id": self.player_id,
            "name": self.name,
            "total_observations": len(self.observation_history),
            "total_actions": len(self.action_history),
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(id={self.player_id}, name={self.name})"
    
    def __repr__(self) -> str:
        """Repr representation of the agent."""
        return self.__str__()

