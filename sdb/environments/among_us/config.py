"""Configuration for simplified Among Us game."""

from dataclasses import dataclass


@dataclass
class AmongUsConfig:
    """Configuration for Among Us game.
    
    This is a simplified version focusing on social deduction elements:
    - No spatial movement
    - Abstract task completion
    - Meetings, discussions, and voting
    - Impostor kills and crewmate tasks
    
    Attributes:
        n_players: Number of players (4-10 recommended)
        n_impostors: Number of impostors
        tasks_per_player: Number of tasks each crewmate must complete
        max_task_rounds: Maximum number of task rounds
        max_discussion_rounds: Maximum discussion statements per meeting
        emergency_meetings: Number of emergency meetings each player can call
        kill_cooldown: Rounds impostors must wait between kills
    """
    n_players: int = 7
    n_impostors: int = 2
    tasks_per_player: int = 3
    max_task_rounds: int = 20
    max_discussion_rounds: int = 10
    emergency_meetings: int = 1  # Per player
    kill_cooldown: int = 2  # Rounds between kills
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_players < 4:
            raise ValueError("Among Us requires at least 4 players")
        if self.n_players > 15:
            raise ValueError("Among Us supports max 15 players")
        
        if self.n_impostors < 1:
            raise ValueError("Need at least 1 impostor")
        if self.n_impostors >= self.n_players // 2:
            raise ValueError("Too many impostors (should be less than half)")
        
        if self.tasks_per_player < 1:
            raise ValueError("tasks_per_player must be at least 1")
        
        if self.max_task_rounds < 1:
            raise ValueError("max_task_rounds must be positive")
        
        if self.kill_cooldown < 1:
            raise ValueError("kill_cooldown must be at least 1")

