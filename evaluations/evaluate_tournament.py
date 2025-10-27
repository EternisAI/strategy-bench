"""Unified tournament evaluator - automatically detects game type and generates tables."""

import sys
import json
from pathlib import Path
from typing import Optional

# Import all evaluators
from evaluations.evaluate_werewolf import WerewolfEvaluator
from evaluations.evaluate_avalon import AvalonEvaluator
from evaluations.evaluate_sheriff import SheriffEvaluator
from evaluations.evaluate_spyfall import SpyfallEvaluator
from evaluations.evaluate_amongus import AmongUsEvaluator
from evaluations.evaluate_secret_hitler import SecretHitlerEvaluator


def detect_game_type(tournament_dir: Path) -> Optional[str]:
    """Detect game type from tournament directory.
    
    Args:
        tournament_dir: Path to tournament directory
        
    Returns:
        Game type string or None if cannot detect
    """
    # Try to read from config file
    config_file = tournament_dir / "tournament_config.yaml"
    if config_file.exists():
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            game = config.get('game')
            if game:
                return game.lower()
    
    # Try to detect from log files
    logs_dir = tournament_dir / "logs"
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.jsonl"))
        if log_files:
            # Read first log file and check events
            with open(log_files[0], 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        data = entry.get('data', {})
                        
                        # Check for game-specific events
                        if 'werewolves' in str(data) or 'seer' in str(data):
                            return 'werewolf'
                        elif 'loyal_servant' in str(data) or 'quest' in str(data).lower():
                            return 'avalon'
                        elif 'sheriff' in str(data) or 'merchant' in str(data):
                            return 'sheriff'
                        elif 'spy' in str(data) and 'location' in str(data):
                            return 'spyfall'
                        elif 'fascist' in str(data) or 'liberal' in str(data):
                            return 'secret_hitler'
                        elif 'imposter' in str(data) or 'task' in str(data):
                            return 'among_us'
    
    # Try to infer from directory name
    dir_name = tournament_dir.name.lower()
    for game in ['werewolf', 'avalon', 'sheriff', 'spyfall', 'secret_hitler', 'among_us']:
        if game in dir_name:
            return game
    
    # Check parent directory
    parent_name = tournament_dir.parent.name.lower()
    for game in ['werewolf', 'avalon', 'sheriff', 'spyfall', 'secret_hitler', 'among_us']:
        if game in parent_name:
            return game
    
    return None


def get_evaluator(game_type: str, tournament_dir: Path):
    """Get appropriate evaluator for game type.
    
    Args:
        game_type: Game type string
        tournament_dir: Tournament directory
        
    Returns:
        Evaluator instance
    """
    evaluators = {
        'werewolf': WerewolfEvaluator,
        'avalon': AvalonEvaluator,
        'sheriff': SheriffEvaluator,
        'spyfall': SpyfallEvaluator,
        'among_us': AmongUsEvaluator,
        'amongus': AmongUsEvaluator,
        'secret_hitler': SecretHitlerEvaluator,
    }
    
    evaluator_class = evaluators.get(game_type)
    if not evaluator_class:
        raise ValueError(f"No evaluator found for game type: {game_type}")
    
    return evaluator_class(tournament_dir)


def main():
    """Run tournament evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate any game tournament and generate summary tables',
        epilog="""
Examples:
  # Auto-detect game type
  python evaluations/evaluate_tournament.py experiments/tournaments/werewolf/20251027_120000
  
  # Specify game type
  python evaluations/evaluate_tournament.py experiments/tournaments/avalon/20251027_120000 --game avalon
  
  # Save to specific output directory
  python evaluations/evaluate_tournament.py experiments/tournaments/sheriff/20251027_120000 --output results/
        """
    )
    parser.add_argument('tournament_dir', type=Path, help='Tournament directory with logs/')
    parser.add_argument('--game', type=str, choices=['werewolf', 'avalon', 'sheriff', 'spyfall', 'secret_hitler', 'among_us'],
                       help='Game type (auto-detect if not specified)')
    parser.add_argument('--output', type=Path, help='Output directory (default: tournament_dir)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate tournament directory
    if not args.tournament_dir.exists():
        print(f"❌ Error: Tournament directory not found: {args.tournament_dir}")
        sys.exit(1)
    
    if not (args.tournament_dir / "logs").exists():
        print(f"❌ Error: No logs/ directory found in: {args.tournament_dir}")
        sys.exit(1)
    
    # Detect or use specified game type
    game_type = args.game
    if not game_type:
        print("🔍 Detecting game type...")
        game_type = detect_game_type(args.tournament_dir)
        if not game_type:
            print("❌ Error: Could not detect game type. Please specify with --game")
            print("   Supported games: werewolf, avalon, sheriff, spyfall")
            sys.exit(1)
        print(f"✓ Detected game type: {game_type}")
    
    # Get evaluator
    try:
        evaluator = get_evaluator(game_type, args.tournament_dir)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print(f"   Note: Evaluator for '{game_type}' may not be implemented yet")
        sys.exit(1)
    
    print(f"\n📊 Evaluating {game_type.upper()} tournament")
    print(f"   Directory: {args.tournament_dir}")
    print()
    
    # Load all games
    try:
        evaluator.load_all_games()
    except Exception as e:
        print(f"❌ Error loading games: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    if not evaluator.results:
        print("❌ No games loaded successfully")
        sys.exit(1)
    
    # Print summary
    evaluator.print_summary()
    
    # Save tables
    try:
        output_dir = args.output if args.output else args.tournament_dir
        evaluator.save_tables(output_dir)
        
        print(f"\n✅ Evaluation complete!")
        print(f"   Summary table: {output_dir}/summary_table.csv")
        print(f"   Detailed stats: {output_dir}/detailed_stats.json")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

