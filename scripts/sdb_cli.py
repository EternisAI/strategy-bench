#!/usr/bin/env python3
"""Command-line interface for Social Deduction Bench."""

import argparse
import sys
import os
import asyncio
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdb.environments.registry import registry, print_registry
from sdb.agents.llm.openrouter_agent import OpenRouterAgent
from sdb.agents.baselines.random_agent import RandomAgent
from sdb.logging.game_logger import GameLogger


# Game configuration mapping
GAME_CONFIGS = {
    "secret_hitler": {
        "env_module": "sdb.environments.secret_hitler",
        "env_class": "SecretHitlerEnv",
        "config_class": "SecretHitlerConfig",
        "config_file": "secret_hitler.yaml",
        "default_players": 7,
        "min_players": 5,
        "max_players": 10,
        "recommended_players": "5-10",
        "async": False,
    },
    "among_us": {
        "env_module": "sdb.environments.among_us",
        "env_class": "AmongUsEnv",
        "config_class": "AmongUsConfig",
        "config_file": "among_us.yaml",
        "default_players": 6,
        "min_players": 6,
        "max_players": 15,
        "recommended_players": "6-10 (2 impostors work best with 6+ players)",
        "async": True,
    },
    "avalon": {
        "env_module": "sdb.environments.avalon",
        "env_class": "AvalonEnv",
        "config_class": "AvalonConfig",
        "config_file": "avalon.yaml",
        "default_players": 5,
        "min_players": 5,
        "max_players": 10,
        "recommended_players": "5-10",
        "async": False,
    },
    "spyfall": {
        "env_module": "sdb.environments.spyfall",
        "env_class": "SpyfallEnv",
        "config_class": "SpyfallConfig",
        "config_file": "spyfall.yaml",
        "default_players": 8,
        "min_players": 3,
        "max_players": 12,
        "recommended_players": "6-10",
        "async": True,
    },
    "werewolf": {
        "env_module": "sdb.environments.werewolf",
        "env_class": "WerewolfEnv",
        "config_class": "WerewolfConfig",
        "config_file": "werewolf.yaml",
        "default_players": 6,
        "min_players": 6,
        "max_players": 20,
        "recommended_players": "6-12 (at least 1 werewolf per 3 villagers)",
        "async": True,
    },
    "sheriff": {
        "env_module": "sdb.environments.sheriff",
        "env_class": "SheriffEnv",
        "config_class": "SheriffConfig",
        "config_file": "sheriff.yaml",
        "default_players": 4,
        "min_players": 3,
        "max_players": 6,
        "recommended_players": "3-5",
        "async": False,
    },
}


def load_game_config(game_name, num_players=None, config_override=None):
    """Load game configuration from YAML or use defaults.
    
    Args:
        game_name: Name of the game
        num_players: Override number of players
        config_override: Dict of config overrides
        
    Returns:
        Tuple of (config_obj, agent_settings, logging_settings)
    """
    if game_name not in GAME_CONFIGS:
        raise ValueError(f"Unknown game: {game_name}")
    
    game_info = GAME_CONFIGS[game_name]
    
    # Import the config class
    module = __import__(game_info["env_module"], fromlist=[game_info["config_class"]])
    ConfigClass = getattr(module, game_info["config_class"])
    
    # Try to load YAML config
    config_path = Path(__file__).parent.parent / "configs" / game_info["config_file"]
    
    if config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
        
        # Extract game config (filter out agent/logging sections)
        game_config = {k: v for k, v in yaml_config.items() 
                      if k not in ['game', 'agent', 'logging']}
        
        # Override num_players if specified
        if num_players:
            game_config['n_players'] = num_players
        
        # Apply any additional overrides
        if config_override:
            game_config.update(config_override)
        
        # Get agent and logging settings
        agent_settings = yaml_config.get('agent', {})
        logging_settings = yaml_config.get('logging', {})
        
        # Create config object
        try:
            config = ConfigClass(**game_config)
        except TypeError as e:
            # If there are extra keys, filter to valid ones
            import inspect
            valid_params = inspect.signature(ConfigClass.__init__).parameters.keys()
            filtered_config = {k: v for k, v in game_config.items() if k in valid_params}
            config = ConfigClass(**filtered_config)
    else:
        # Use defaults
        config_kwargs = {'n_players': num_players or game_info["default_players"]}
        config = ConfigClass(**config_kwargs)
        agent_settings = {}
        logging_settings = {}
    
    return config, agent_settings, logging_settings


def create_agents(num_players, agent_type, model, temperature, agent_settings):
    """Create agents for the game.
    
    Args:
        num_players: Number of agents to create
        agent_type: "llm" or "random"
        model: LLM model name (for llm agents)
        temperature: Temperature setting (for llm agents)
        agent_settings: Additional settings from YAML
        
    Returns:
        List of agents
    """
    agents = []
    
    for i in range(num_players):
        if agent_type == "llm":
            # Use CLI args, fallback to YAML settings, then defaults
            agent_model = model or agent_settings.get('model', 'openai/gpt-4o-mini')
            agent_temp = temperature if temperature is not None else agent_settings.get('temperature', 0.8)
            agent_memory = agent_settings.get('memory_capacity', 35)
            
            agent = OpenRouterAgent(
                player_id=i,
                model=agent_model,
                temperature=agent_temp,
                memory_capacity=agent_memory
            )
        else:  # random
            agent = RandomAgent(player_id=i, seed=i)
        
        agents.append(agent)
    
    return agents


async def run_game_async(env):
    """Run an async game."""
    result = await env.play_game()
    return result


def run_game_sync(env):
    """Run a sync game."""
    result = env.play_game()
    return result


def validate_player_count(game_name, num_players):
    """Validate player count for a game and provide helpful warnings.
    
    Args:
        game_name: Name of the game
        num_players: Number of players
        
    Returns:
        Tuple of (is_valid, warning_message)
    """
    if game_name not in GAME_CONFIGS:
        return False, f"Unknown game: {game_name}"
    
    info = GAME_CONFIGS[game_name]
    min_p = info.get("min_players", 1)
    max_p = info.get("max_players", 100)
    recommended = info.get("recommended_players", "")
    
    if num_players < min_p:
        return False, f"{game_name} requires at least {min_p} players (you specified {num_players})"
    
    if num_players > max_p:
        return False, f"{game_name} supports at most {max_p} players (you specified {num_players})"
    
    # Game-specific warnings
    warnings = []
    
    if game_name == "among_us":
        # Among Us needs enough players for impostors
        if num_players < 6:
            warnings.append(f"Among Us works best with 6+ players (2 impostors). With {num_players} players, you may need to adjust n_impostors in the config.")
    
    if game_name == "werewolf":
        # Werewolf needs balanced roles
        if num_players < 6:
            warnings.append(f"Werewolf is more balanced with 6+ players.")
    
    if game_name == "sheriff":
        # Sheriff is best with fewer players
        if num_players > 5:
            warnings.append(f"Sheriff of Nottingham works best with 3-5 players. With {num_players}, games may be longer.")
    
    if warnings:
        return True, " ".join(warnings)
    
    return True, None


def cmd_list_environments(args):
    """List all available environments."""
    print("\nAvailable Games:")
    print("=" * 70)
    for game_name, info in GAME_CONFIGS.items():
        default_players = info["default_players"]
        recommended = info.get("recommended_players", f"{default_players}")
        game_type = "Async" if info["async"] else "Sync"
        print(f"  • {game_name:<20} Players: {recommended:<40}")
        print(f"    {'':22} (Default: {default_players}, {game_type})")
    print()


def cmd_play_game(args):
    """Play a single game."""
    # Check API key for LLM agents
    if args.agent_type == "llm" and not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    game_name = args.environment
    
    if game_name not in GAME_CONFIGS:
        print(f"Error: Unknown game '{game_name}'")
        print(f"Available games: {', '.join(GAME_CONFIGS.keys())}")
        return
    
    game_info = GAME_CONFIGS[game_name]
    
    print(f"\nPlaying: {game_name}")
    print("=" * 70)
    
    # Load configuration
    try:
        config, agent_settings, logging_settings = load_game_config(
            game_name,
            num_players=args.num_players,
            config_override={}
        )
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Validate player count
    is_valid, warning = validate_player_count(game_name, config.n_players)
    if not is_valid:
        print(f"Error: {warning}")
        print(f"Recommended: {GAME_CONFIGS[game_name]['recommended_players']}")
        return
    
    print(f"Configuration:")
    print(f"  • Players: {config.n_players}")
    
    # Show warning if any
    if warning:
        print(f"  ⚠️  Warning: {warning}")
    
    print(f"  • Agent type: {args.agent_type}")
    
    # Create agents
    agents = create_agents(
        config.n_players,
        args.agent_type,
        args.model,
        args.temperature,
        agent_settings
    )
    
    if args.agent_type == "llm":
        model_name = args.model or agent_settings.get('model', 'openai/gpt-4o-mini')
        print(f"  • Model: {model_name}")
    
    # Setup logging
    output_dir = Path(args.output_dir or logging_settings.get('output_dir', f'experiments/{game_name}_cli'))
    logger = GameLogger(output_dir=output_dir, log_private=True)
    print(f"  • Logs: {output_dir}/ (including private info)")
    
    # Import and create environment
    module = __import__(game_info["env_module"], fromlist=[game_info["env_class"]])
    EnvClass = getattr(module, game_info["env_class"])
    
    env = EnvClass(agents=agents, config=config, logger=logger)
    
    # Run game
    print(f"\nStarting game...")
    print("-" * 70)
    
    try:
        if game_info["async"]:
            result = asyncio.run(run_game_async(env))
        else:
            result = run_game_sync(env)
        
        # Display results
        print("\n" + "=" * 70)
        print("GAME OVER")
        print("=" * 70)
        
        if hasattr(result, 'winner'):
            winner = result.winner
            if winner:
                print(f"Winner: {winner}")
            else:
                print("Winner: Tie/Timeout")
        
        if hasattr(result, 'win_reason'):
            print(f"Reason: {result.win_reason}")
        
        if hasattr(result, 'num_rounds'):
            print(f"Rounds: {result.num_rounds}")
        elif hasattr(result, 'duration'):
            print(f"Duration: {result.duration}")
        
        print(f"\nLog saved to: {logger.log_file}")
        
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user")
    except Exception as e:
        print(f"\nError during game: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Social Deduction Bench - Benchmark LLMs in social deduction games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available games
  python scripts/sdb_cli.py list
  
  # Play Secret Hitler with 7 LLM players
  python scripts/sdb_cli.py play secret_hitler --num-players 7
  
  # Play Among Us with random agents
  python scripts/sdb_cli.py play among_us --agent-type random
  
  # Play Avalon with a specific model
  python scripts/sdb_cli.py play avalon --model anthropic/claude-3.5-sonnet
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List environments
    parser_list = subparsers.add_parser("list", help="List available game environments")
    parser_list.set_defaults(func=cmd_list_environments)
    
    # Play single game
    parser_play = subparsers.add_parser("play", help="Play a single game")
    parser_play.add_argument("environment", 
                           choices=list(GAME_CONFIGS.keys()),
                           help="Game to play")
    parser_play.add_argument("--num-players", type=int, 
                           help="Number of players (default: game-specific)")
    parser_play.add_argument("--agent-type", choices=["llm", "random"], default="llm",
                           help="Agent type (default: llm)")
    parser_play.add_argument("--model", 
                           help="LLM model (default: from config or openai/gpt-4o-mini)")
    parser_play.add_argument("--temperature", type=float,
                           help="Temperature for LLM (default: from config or 0.8)")
    parser_play.add_argument("--output-dir",
                           help="Output directory for logs")
    parser_play.set_defaults(func=cmd_play_game)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
