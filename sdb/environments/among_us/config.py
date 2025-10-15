"""Configuration for simplified Among Us game."""

from dataclasses import dataclass


@dataclass
class AmongUsConfig:
    """Configuration for Among Us game with spatial mechanics.
    
    Features spatial movement, room-based tasks, and social deduction:
    - Spatial movement via corridors and vents
    - Room-specific task assignment
    - Visibility limited to same room
    - Meetings, discussions, and voting
    - Impostor kills and crewmate tasks
    
    Attributes:
        n_players: Number of players (4-15)
        n_impostors: Number of impostors
        tasks_per_player: Number of tasks each crewmate must complete
        max_task_rounds: Maximum number of task rounds (original: 50 for 7-player)
        discussion_rounds: Discussion rounds per meeting (original: 3)
        emergency_meetings: Number of emergency meetings each player can call
        kill_cooldown: Rounds impostors must wait between kills (original: 3)
    """
    n_players: int = 7
    n_impostors: int = 2
    tasks_per_player: int = 3
    max_task_rounds: int = 50  # Match original (was 20)
    discussion_rounds: int = 3  # Match original (was max_discussion_rounds: 10)
    emergency_meetings: int = 1  # Per player
    kill_cooldown: int = 3  # Match original (was 2)
    
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

