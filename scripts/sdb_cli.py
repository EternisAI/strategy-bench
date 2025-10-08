#!/usr/bin/env python3
"""Command-line interface for Social Deduction Bench."""

import argparse
import sys
import os
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdb.environments.registry import registry, print_registry
from sdb.agents.llm import OpenRouterAgent
from sdb.agents.baselines import RandomAgent
from sdb.tournament import TournamentManager, TournamentConfig
from sdb.evaluation import Evaluator
from sdb.analysis import Visualizer


def cmd_list_environments(args):
    """List all available environments."""
    print_registry()


def cmd_play_game(args):
    """Play a single game."""
    print(f"\n🎮 Playing {args.environment}")
    print(f"   Players: {args.num_players}")
    print(f"   Model: {args.model}")
    
    # Get environment
    env_class = registry.get(args.environment)
    
    # Create agents
    agents = []
    for i in range(args.num_players):
        if args.agent_type == "llm":
            agent = OpenRouterAgent(
                player_id=i,
                name=f"Agent_{i}",
                model=args.model,
                temperature=args.temperature,
                memory_capacity=50
            )
        else:
            agent = RandomAgent(player_id=i, name=f"Random_{i}")
        agents.append(agent)
    
    # Import environment-specific config
    logger = None
    if args.environment == "secret_hitler":
        from sdb.environments.secret_hitler import SecretHitlerConfig
        from sdb.logging import GameLogger
        
        config = SecretHitlerConfig(n_players=args.num_players, log_private_info=True)
        logger = GameLogger(game_id=f"{args.environment}_cli", output_dir=args.output_dir)
        env = env_class(agents=agents, config=config, logger=logger)
    else:
        env = env_class(agents=agents)
    
    # Run game (synchronous for Secret Hitler)
    print("\n▶️  Playing game...\n")
    result = env.play_game()
    
    print(f"\n🏆 Winner: {result.winner}")
    print(f"   Reason: {result.win_reason}")
    print(f"   Rounds: {result.num_rounds}")
    
    if logger and hasattr(logger, 'get_log_file'):
        print(f"   Log: {logger.get_log_file()}")


def cmd_run_tournament(args):
    """Run a tournament."""
    print(f"\n🏆 Running {args.tournament_type} tournament")
    print(f"   Environment: {args.environment}")
    print(f"   Games: {args.num_games}")
    print(f"   Output: {args.output_dir}")
    
    # Create agents
    agents = []
    num_players = args.num_players
    
    for i in range(num_players):
        if args.agent_type == "llm":
            agent = OpenRouterAgent(
                player_id=i,
                name=f"LLM_{i}",
                model=args.model
            )
        else:
            agent = RandomAgent(player_id=i, name=f"Random_{i}")
        agents.append(agent)
    
    # Create tournament config
    config = TournamentConfig(
        name=args.name,
        environment=args.environment,
        num_games=args.num_games,
        output_dir=Path(args.output_dir),
        log_games=True,
        seed=args.seed
    )
    
    # Run tournament
    manager = TournamentManager(output_dir=Path(args.output_dir))
    result = asyncio.run(manager.run_tournament(args.tournament_type, config, agents))
    
    print(f"\n✅ Tournament complete!")
    print(f"   Results saved to: {args.output_dir}")


def cmd_evaluate(args):
    """Evaluate game logs."""
    print(f"\n📊 Evaluating: {args.input}")
    
    evaluator = Evaluator()
    
    input_path = Path(args.input)
    if input_path.is_file():
        # Single game log
        if input_path.suffix == ".jsonl":
            metrics = evaluator.evaluate_game_log(input_path)
        elif input_path.suffix == ".json":
            metrics = evaluator.evaluate_tournament(input_path)
        else:
            print(f"❌ Unknown file type: {input_path.suffix}")
            return
    else:
        print(f"❌ Path not found: {input_path}")
        return
    
    # Save summary
    output_path = Path(args.output) if args.output else input_path.parent / "metrics.json"
    evaluator.save_summary(output_path)
    
    print(f"✅ Evaluation complete!")
    print(f"   Metrics saved to: {output_path}")


def cmd_visualize(args):
    """Create visualizations."""
    print(f"\n📈 Creating visualizations")
    print(f"   Tournament: {args.tournament}")
    print(f"   Metrics: {args.metrics}")
    
    visualizer = Visualizer(output_dir=Path(args.output_dir))
    dashboard = visualizer.create_dashboard(
        Path(args.tournament),
        Path(args.metrics)
    )
    
    print(f"✅ Visualizations created!")
    print(f"   Dashboard saved to: {args.output_dir}/dashboard.json")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Social Deduction Bench - Benchmark LLMs in social deduction games",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List environments
    parser_list = subparsers.add_parser("list", help="List available game environments")
    parser_list.set_defaults(func=cmd_list_environments)
    
    # Play single game
    parser_play = subparsers.add_parser("play", help="Play a single game")
    parser_play.add_argument("environment", help="Game environment name")
    parser_play.add_argument("--num-players", type=int, default=5, help="Number of players")
    parser_play.add_argument("--agent-type", choices=["llm", "random"], default="llm")
    parser_play.add_argument("--model", default="openai/gpt-4o-mini", help="LLM model")
    parser_play.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser_play.add_argument("--output-dir", default="experiments/cli_games", help="Output directory")
    parser_play.set_defaults(func=cmd_play_game)
    
    # Run tournament
    parser_tournament = subparsers.add_parser("tournament", help="Run a tournament")
    parser_tournament.add_argument("environment", help="Game environment")
    parser_tournament.add_argument("--name", default="tournament", help="Tournament name")
    parser_tournament.add_argument("--tournament-type", choices=["round_robin", "swiss"], 
                                   default="round_robin")
    parser_tournament.add_argument("--num-games", type=int, default=10, help="Number of games")
    parser_tournament.add_argument("--num-players", type=int, default=5, help="Number of players")
    parser_tournament.add_argument("--agent-type", choices=["llm", "random"], default="llm")
    parser_tournament.add_argument("--model", default="openai/gpt-4o-mini", help="LLM model")
    parser_tournament.add_argument("--output-dir", default="tournaments", help="Output directory")
    parser_tournament.add_argument("--seed", type=int, help="Random seed")
    parser_tournament.set_defaults(func=cmd_run_tournament)
    
    # Evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate game logs")
    parser_eval.add_argument("input", help="Path to game log or tournament result")
    parser_eval.add_argument("--output", help="Output path for metrics")
    parser_eval.set_defaults(func=cmd_evaluate)
    
    # Visualize
    parser_viz = subparsers.add_parser("visualize", help="Create visualizations")
    parser_viz.add_argument("--tournament", required=True, help="Tournament result JSON")
    parser_viz.add_argument("--metrics", required=True, help="Metrics JSON")
    parser_viz.add_argument("--output-dir", default="visualizations", help="Output directory")
    parser_viz.set_defaults(func=cmd_visualize)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()

