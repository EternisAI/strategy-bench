"""Base environment class for all game implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import time

from sdb.core.base_state import BaseState
from sdb.core.base_agent import BaseAgent
from sdb.core.types import Action, GameResult, PlayerID
from sdb.core.exceptions import EnvironmentError, InvalidActionError
from sdb.core.utils import generate_game_id


class BaseEnvironment(ABC):
    """Abstract base class for all game environments.
    
    All game implementations should inherit from this class and implement
    the abstract methods. This provides a consistent interface for running
    games, tournaments, and evaluations.
    """
    
    def __init__(
        self,
        agents: List[BaseAgent],
        config: Optional[Dict[str, Any]] = None,
        game_id: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize environment.
        
        Args:
            agents: List of agents playing the game
            config: Configuration dictionary for the environment
            game_id: Unique identifier for this game (auto-generated if None)
            seed: Random seed for reproducibility
        """
        self.agents = agents
        self.num_players = len(agents)
        self.config = config or {}
        self.game_id = game_id or generate_game_id(self.__class__.__name__.lower())
        self.seed = seed
        
        # Validate number of players
        self._validate_num_players()
        
        # Game state
        self.state: Optional[BaseState] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Initialize game
        self.reset()
    
    @abstractmethod
    def reset(self) -> BaseState:
        """Reset the environment to start a new game.
        
        Returns:
            Initial game state
        """
        pass
    
    @abstractmethod
    def step(self, action: Action) -> Tuple[BaseState, bool]:
        """Execute one step of the game.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (new_state, is_terminal)
            
        Raises:
            InvalidActionError: If action is invalid
            EnvironmentError: If execution fails
        """
        pass
    
    @abstractmethod
    def get_winner(self) -> Optional[Any]:
        """Determine the winner(s) of the game.
        
        Returns:
            Winner identifier (team name, player ID, list of IDs, etc.)
            Returns None if game is not finished
        """
        pass
    
    @abstractmethod
    def get_win_reason(self) -> Optional[str]:
        """Get the reason for the win condition.
        
        Returns:
            String describing why the game ended
            Returns None if game is not finished
        """
        pass
    
    def play_game(self) -> GameResult:
        """Play a complete game from start to finish.
        
        Returns:
            GameResult object with game outcome and statistics
        """
        self.start_time = time.time()
        self.reset()
        
        round_count = 0
        max_rounds = self.config.get("max_rounds", 1000)
        
        while not self.state.is_terminal and round_count < max_rounds:
            # Get current player
            current_player_id = self._get_current_player()
            current_agent = self.agents[current_player_id]
            
            # Get observation
            observation = self.state.get_observation(current_player_id)
            current_agent.observe(observation)
            
            # Agent chooses action
            try:
                action = current_agent.act(observation)
                current_agent.record_action(action)
            except Exception as e:
                # Handle agent failures gracefully
                action = self._get_fallback_action(current_player_id)
                if action is None:
                    raise EnvironmentError(
                        f"Agent {current_player_id} failed and no fallback available",
                        details={"error": str(e)}
                    )
            
            # Execute action
            self.state, is_terminal = self.step(action)
            
            round_count += 1
        
        self.end_time = time.time()
        
        # Construct result
        return self._build_game_result()
    
    async def play_game_async(self) -> GameResult:
        """Play a complete game asynchronously.
        
        For games that support async execution (e.g., with LLM agents).
        Default implementation calls synchronous play_game().
        
        Returns:
            GameResult object with game outcome and statistics
        """
        return self.play_game()
    
    @abstractmethod
    def _get_current_player(self) -> PlayerID:
        """Get the ID of the current player whose turn it is.
        
        Returns:
            Player ID
        """
        pass
    
    @abstractmethod
    def _validate_num_players(self) -> None:
        """Validate that the number of players is valid for this game.
        
        Raises:
            EnvironmentError: If number of players is invalid
        """
        pass
    
    def _get_fallback_action(self, player_id: PlayerID) -> Optional[Action]:
        """Get a fallback action when an agent fails.
        
        Override this in subclasses to provide game-specific fallbacks.
        
        Args:
            player_id: ID of the player who needs a fallback action
            
        Returns:
            Fallback action or None
        """
        return None
    
    def _build_game_result(self) -> GameResult:
        """Build GameResult object from current game state.
        
        Returns:
            GameResult object
        """
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0.0
        
        # Collect player stats
        player_stats = {}
        for agent in self.agents:
            player_stats[agent.player_id] = agent.get_stats()
        
        return GameResult(
            game_id=self.game_id,
            winner=self.get_winner(),
            win_reason=self.get_win_reason() or "Unknown",
            num_rounds=self.state.round_number,
            duration_seconds=duration,
            player_stats=player_stats,
            metadata=self.config,
        )
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the current game state.
        
        Args:
            mode: Render mode ('human', 'ansi', 'rgb_array', etc.)
            
        Returns:
            String representation or None
        """
        if mode == "human":
            return str(self.state)
        return None
    
    def __str__(self) -> str:
        """String representation of the environment."""
        return f"{self.__class__.__name__}(game_id={self.game_id}, players={self.num_players})"

