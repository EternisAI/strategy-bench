"""Random agent that chooses actions randomly."""

import random
from typing import Any, Dict, Optional

from sdb.core.base_agent import BaseAgent
from sdb.core.types import Action, Observation, ActionType


class RandomAgent(BaseAgent):
    """Agent that chooses actions randomly.
    
    Useful as a baseline for evaluation.
    """
    
    def __init__(
        self,
        player_id: int,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize random agent.
        
        Args:
            player_id: Unique identifier
            name: Agent name
            seed: Random seed for reproducibility
            config: Additional configuration
        """
        super().__init__(player_id=player_id, name=name, config=config)
        self.rng = random.Random(seed)
        self.legal_actions = []  # Will be set by environment
    
    def act(self, observation: Observation) -> Action:
        """Choose a random action.
        
        Args:
            observation: Current observation
            
        Returns:
            Randomly chosen action
        """
        # If legal actions provided in observation, use those
        legal_actions = observation.data.get("legal_actions", [])
        
        if legal_actions:
            # Choose from provided legal actions
            chosen = self.rng.choice(legal_actions)
            if isinstance(chosen, Action):
                return chosen
            elif isinstance(chosen, dict):
                return Action(**chosen)
        
        # Otherwise, return a generic SPEAK action
        return Action(
            player_id=self.player_id,
            action_type=ActionType.SPEAK,
            data={"message": "Random action"}
        )
    
    def set_legal_actions(self, actions):
        """Set the list of legal actions.
        
        Args:
            actions: List of legal actions
        """
        self.legal_actions = actions

