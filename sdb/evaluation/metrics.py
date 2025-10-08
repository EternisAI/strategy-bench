"""Metrics for evaluating agent performance."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GameMetrics:
    """Basic game performance metrics."""
    
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    average_game_length: float = 0.0
    average_survival_rounds: float = 0.0
    
    def update(self, won: bool, game_length: int, survival_rounds: int) -> None:
        """Update metrics with new game result.
        
        Args:
            won: Whether the player won
            game_length: Number of rounds in the game
            survival_rounds: Number of rounds player survived
        """
        self.games_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
        
        # Update averages
        self.win_rate = self.wins / self.games_played
        
        # Running average for game length
        self.average_game_length = (
            (self.average_game_length * (self.games_played - 1) + game_length)
            / self.games_played
        )
        
        # Running average for survival
        self.average_survival_rounds = (
            (self.average_survival_rounds * (self.games_played - 1) + survival_rounds)
            / self.games_played
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "average_game_length": self.average_game_length,
            "average_survival_rounds": self.average_survival_rounds,
        }


@dataclass
class DeceptionMetrics:
    """Metrics for measuring deception and social reasoning."""
    
    successful_lies: int = 0
    caught_lies: int = 0
    correct_accusations: int = 0
    incorrect_accusations: int = 0
    deception_success_rate: float = 0.0
    accusation_accuracy: float = 0.0
    trust_score: float = 0.5  # How much other players trust this agent
    
    def update_deception(self, successful: bool) -> None:
        """Update deception metrics.
        
        Args:
            successful: Whether the deception was successful
        """
        if successful:
            self.successful_lies += 1
        else:
            self.caught_lies += 1
        
        total = self.successful_lies + self.caught_lies
        self.deception_success_rate = self.successful_lies / total if total > 0 else 0.0
    
    def update_accusation(self, correct: bool) -> None:
        """Update accusation metrics.
        
        Args:
            correct: Whether the accusation was correct
        """
        if correct:
            self.correct_accusations += 1
        else:
            self.incorrect_accusations += 1
        
        total = self.correct_accusations + self.incorrect_accusations
        self.accusation_accuracy = self.correct_accusations / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "successful_lies": self.successful_lies,
            "caught_lies": self.caught_lies,
            "deception_success_rate": self.deception_success_rate,
            "correct_accusations": self.correct_accusations,
            "incorrect_accusations": self.incorrect_accusations,
            "accusation_accuracy": self.accusation_accuracy,
            "trust_score": self.trust_score,
        }


@dataclass
class CommunicationMetrics:
    """Metrics for measuring communication effectiveness."""
    
    messages_sent: int = 0
    average_message_length: float = 0.0
    information_shared: int = 0
    persuasion_attempts: int = 0
    successful_persuasions: int = 0
    persuasion_success_rate: float = 0.0
    
    def update_message(self, message_length: int, shared_info: bool = False) -> None:
        """Update communication metrics.
        
        Args:
            message_length: Length of message in characters
            shared_info: Whether information was shared
        """
        self.messages_sent += 1
        self.average_message_length = (
            (self.average_message_length * (self.messages_sent - 1) + message_length)
            / self.messages_sent
        )
        
        if shared_info:
            self.information_shared += 1
    
    def update_persuasion(self, successful: bool) -> None:
        """Update persuasion metrics.
        
        Args:
            successful: Whether persuasion was successful
        """
        self.persuasion_attempts += 1
        if successful:
            self.successful_persuasions += 1
        
        self.persuasion_success_rate = (
            self.successful_persuasions / self.persuasion_attempts
            if self.persuasion_attempts > 0 else 0.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "messages_sent": self.messages_sent,
            "average_message_length": self.average_message_length,
            "information_shared": self.information_shared,
            "persuasion_attempts": self.persuasion_attempts,
            "successful_persuasions": self.successful_persuasions,
            "persuasion_success_rate": self.persuasion_success_rate,
        }


@dataclass
class AgentMetrics:
    """Combined metrics for an agent."""
    
    agent_id: int
    agent_name: str
    game_metrics: GameMetrics = field(default_factory=GameMetrics)
    deception_metrics: DeceptionMetrics = field(default_factory=DeceptionMetrics)
    communication_metrics: CommunicationMetrics = field(default_factory=CommunicationMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "game_metrics": self.game_metrics.to_dict(),
            "deception_metrics": self.deception_metrics.to_dict(),
            "communication_metrics": self.communication_metrics.to_dict(),
        }


def calculate_elo_rating(
    player_rating: float,
    opponent_rating: float,
    player_won: bool,
    k_factor: float = 32.0
) -> float:
    """Calculate new ELO rating after a game.
    
    Args:
        player_rating: Current player rating
        opponent_rating: Current opponent rating
        player_won: Whether player won
        k_factor: K-factor for rating adjustment
        
    Returns:
        New player rating
    """
    expected_score = 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))
    actual_score = 1.0 if player_won else 0.0
    new_rating = player_rating + k_factor * (actual_score - expected_score)
    return new_rating


def calculate_team_elo(
    team_ratings: List[float],
    opponent_ratings: List[float],
    team_won: bool,
    k_factor: float = 32.0
) -> List[float]:
    """Calculate new ELO ratings for team game.
    
    Args:
        team_ratings: Current ratings for team members
        opponent_ratings: Current ratings for opponents
        team_won: Whether team won
        k_factor: K-factor for rating adjustment
        
    Returns:
        New ratings for team members
    """
    avg_team = np.mean(team_ratings)
    avg_opponent = np.mean(opponent_ratings)
    
    expected_score = 1 / (1 + 10 ** ((avg_opponent - avg_team) / 400))
    actual_score = 1.0 if team_won else 0.0
    
    new_ratings = []
    for rating in team_ratings:
        new_rating = rating + k_factor * (actual_score - expected_score)
        new_ratings.append(new_rating)
    
    return new_ratings

