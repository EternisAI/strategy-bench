"""Environment registry for easy access to all game environments."""

from typing import Dict, Type, List, Optional, Any
from pathlib import Path

from sdb.core.base_env import BaseEnvironment


class EnvironmentRegistry:
    """Registry for all available game environments.
    
    Provides centralized access to game environments and their metadata.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._environments: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        env_class: Type[BaseEnvironment],
        description: str,
        min_players: int,
        max_players: int,
        difficulty: str = "medium",
        tags: Optional[List[str]] = None,
        **metadata
    ) -> None:
        """Register a new environment.
        
        Args:
            name: Unique identifier for the environment
            env_class: Environment class (must inherit from BaseEnvironment)
            description: Brief description of the game
            min_players: Minimum number of players
            max_players: Maximum number of players
            difficulty: Difficulty level (easy/medium/hard)
            tags: List of tags for categorization
            **metadata: Additional metadata
        """
        if not issubclass(env_class, BaseEnvironment):
            raise ValueError(f"{env_class} must inherit from BaseEnvironment")
        
        self._environments[name] = {
            "class": env_class,
            "description": description,
            "min_players": min_players,
            "max_players": max_players,
            "difficulty": difficulty,
            "tags": tags or [],
            "metadata": metadata
        }
    
    def get(self, name: str) -> Type[BaseEnvironment]:
        """Get environment class by name.
        
        Args:
            name: Environment name
            
        Returns:
            Environment class
            
        Raises:
            KeyError: If environment not found
        """
        if name not in self._environments:
            raise KeyError(f"Environment '{name}' not found. Available: {self.list_names()}")
        return self._environments[name]["class"]
    
    def get_info(self, name: str) -> Dict[str, Any]:
        """Get full information about an environment.
        
        Args:
            name: Environment name
            
        Returns:
            Dictionary with environment metadata
        """
        if name not in self._environments:
            raise KeyError(f"Environment '{name}' not found")
        return self._environments[name].copy()
    
    def list_names(self) -> List[str]:
        """Get list of all registered environment names.
        
        Returns:
            List of environment names
        """
        return list(self._environments.keys())
    
    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered environments.
        
        Returns:
            Dictionary mapping names to environment info
        """
        return {name: self.get_info(name) for name in self._environments}
    
    def filter_by_players(self, num_players: int) -> List[str]:
        """Get environments that support a specific number of players.
        
        Args:
            num_players: Number of players
            
        Returns:
            List of environment names
        """
        return [
            name for name, info in self._environments.items()
            if info["min_players"] <= num_players <= info["max_players"]
        ]
    
    def filter_by_tag(self, tag: str) -> List[str]:
        """Get environments with a specific tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of environment names
        """
        return [
            name for name, info in self._environments.items()
            if tag in info["tags"]
        ]
    
    def filter_by_difficulty(self, difficulty: str) -> List[str]:
        """Get environments by difficulty level.
        
        Args:
            difficulty: Difficulty level (easy/medium/hard)
            
        Returns:
            List of environment names
        """
        return [
            name for name, info in self._environments.items()
            if info["difficulty"] == difficulty
        ]


# Global registry instance
registry = EnvironmentRegistry()


# Register Secret Hitler
from sdb.environments.secret_hitler import SecretHitlerEnv

registry.register(
    name="secret_hitler",
    env_class=SecretHitlerEnv,
    description="Secret Hitler: A social deduction game of hidden identities and political intrigue",
    min_players=5,
    max_players=10,
    difficulty="hard",
    tags=["deduction", "voting", "hidden_role", "team_game", "policy", "deception"],
    complexity="high",
    game_length="medium",
    deception_level="high",
    communication="open"
)


def get_env(name: str) -> Type[BaseEnvironment]:
    """Get environment class by name.
    
    Args:
        name: Environment name
        
    Returns:
        Environment class
    """
    return registry.get(name)


def list_environments() -> List[str]:
    """Get list of all available environments.
    
    Returns:
        List of environment names
    """
    return registry.list_names()


def print_registry() -> None:
    """Print information about all registered environments."""
    print("\n" + "="*80)
    print("🎮 REGISTERED GAME ENVIRONMENTS")
    print("="*80)
    
    for name, info in registry.list_all().items():
        print(f"\n📦 {name}")
        print(f"   {info['description']}")
        print(f"   Players: {info['min_players']}-{info['max_players']}")
        print(f"   Difficulty: {info['difficulty']}")
        print(f"   Tags: {', '.join(info['tags'])}")


if __name__ == "__main__":
    # Demo the registry
    print_registry()

