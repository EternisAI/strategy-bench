"""Batch evaluate multiple tournaments at once."""

import sys
from pathlib import Path
from typing import List
import pandas as pd

from evaluations.evaluate_tournament import detect_game_type, get_evaluator


def find_tournaments(base_dir: Path, game_type: str = None) -> List[Path]:
    """Find all tournament directories.
    
    Args:
        base_dir: Base directory to search (e.g., experiments/tournaments)
        game_type: Optional game type filter
        
    Returns:
        List of tournament directory paths
    """
    tournaments = []
    
    if game_type:
        # Search in specific game directory
        game_dir = base_dir / game_type
        if game_dir.exists():
            for subdir in game_dir.iterdir():
                if subdir.is_dir() and (subdir / "logs").exists():
                    tournaments.append(subdir)
    else:
        # Search all subdirectories
        for game_dir in base_dir.iterdir():
            if not game_dir.is_dir():
                continue
            for subdir in game_dir.iterdir():
                if subdir.is_dir() and (subdir / "logs").exists():
                    tournaments.append(subdir)
    
    return sorted(tournaments)


def evaluate_all_tournaments(tournament_dirs: List[Path], output_dir: Path = None):
    """Evaluate all tournaments and generate combined report.
    
    Args:
        tournament_dirs: List of tournament directories
        output_dir: Optional output directory for combined results
    """
    all_results = []
    
    print(f"Found {len(tournament_dirs)} tournaments to evaluate\n")
    
    for i, tournament_dir in enumerate(tournament_dirs, 1):
        print(f"[{i}/{len(tournament_dirs)}] Evaluating: {tournament_dir.name}")
        
        # Detect game type
        game_type = detect_game_type(tournament_dir)
        if not game_type:
            print(f"  ⚠️  Could not detect game type, skipping")
            continue
        
        print(f"  Game: {game_type}")
        
        # Get evaluator
        try:
            evaluator = get_evaluator(game_type, tournament_dir)
        except ValueError as e:
            print(f"  ⚠️  {e}, skipping")
            continue
        
        # Load and evaluate
        try:
            evaluator.load_all_games()
            if not evaluator.results:
                print(f"  ⚠️  No games loaded, skipping")
                continue
            
            # Generate summary
            summary_df = evaluator.generate_summary_table()
            summary_df['Tournament'] = tournament_dir.name
            summary_df['Game'] = game_type
            
            all_results.append(summary_df)
            
            # Save individual results
            evaluator.save_tables()
            print(f"  ✓ Evaluated {len(evaluator.results)} games")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
        
        print()
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns
        cols = ['Tournament', 'Game', 'Model', 'Games', 'Wins', 'Win Rate', 'Avg Rounds']
        other_cols = [c for c in combined_df.columns if c not in cols]
        combined_df = combined_df[cols + other_cols]
        
        # Save combined results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "combined_results.csv"
        else:
            output_file = Path("combined_tournament_results.csv")
        
        combined_df.to_csv(output_file, index=False)
        print(f"\n✅ Combined results saved to: {output_file}")
        print(f"   Total tournaments: {len(all_results)}")
        print(f"   Total rows: {len(combined_df)}")
        
        # Print summary
        print("\n" + "="*80)
        print("COMBINED TOURNAMENT SUMMARY")
        print("="*80)
        
        # Group by model across all tournaments
        model_summary = combined_df.groupby('Model').agg({
            'Games': 'sum',
            'Wins': 'sum',
        })
        model_summary['Overall Win Rate'] = (
            model_summary['Wins'] / model_summary['Games'] * 100
        ).map('{:.1f}%'.format)
        
        print(model_summary.to_string())
        print("="*80)


def main():
    """Run batch evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch evaluate multiple tournaments',
        epilog="""
Examples:
  # Evaluate all werewolf tournaments
  python evaluations/batch_evaluate.py experiments/tournaments --game werewolf
  
  # Evaluate all tournaments
  python evaluations/batch_evaluate.py experiments/tournaments
  
  # Save to custom directory
  python evaluations/batch_evaluate.py experiments/tournaments --output batch_results/
        """
    )
    parser.add_argument('base_dir', type=Path, help='Base tournaments directory')
    parser.add_argument('--game', type=str, help='Filter by game type')
    parser.add_argument('--output', type=Path, help='Output directory for combined results')
    
    args = parser.parse_args()
    
    # Validate base directory
    if not args.base_dir.exists():
        print(f"❌ Error: Base directory not found: {args.base_dir}")
        sys.exit(1)
    
    # Find tournaments
    print(f"🔍 Searching for tournaments in: {args.base_dir}")
    if args.game:
        print(f"   Filtering by game: {args.game}")
    print()
    
    tournament_dirs = find_tournaments(args.base_dir, args.game)
    
    if not tournament_dirs:
        print("❌ No tournaments found")
        sys.exit(1)
    
    # Evaluate all
    evaluate_all_tournaments(tournament_dirs, args.output)


if __name__ == '__main__':
    main()

