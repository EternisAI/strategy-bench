#!/usr/bin/env python3
"""Generic tournament runner that executes games from configuration files."""

import argparse
import asyncio
import csv
import sys
import json
import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback
import importlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdb.agents.llm.openrouter_agent import OpenRouterAgent
from sdb.logging.game_logger import GameLogger
from enum import Enum


class TournamentJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for tournament results."""
    
    def default(self, obj):
        """Convert non-serializable objects to JSON-compatible format."""
        # Handle Enums
        if isinstance(obj, Enum):
            return obj.value if hasattr(obj, 'value') else str(obj)
        
        # Handle datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
        
        # Try to convert to dict if possible
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        
        # Default behavior
        return super().default(obj)


class TournamentProgressTracker:
    """Tracks and displays tournament progress in real-time."""
    
    def __init__(self, total_games: int, max_concurrent: int, output_dir: Path):
        """Initialize progress tracker.
        
        Args:
            total_games: Total number of games in tournament
            max_concurrent: Maximum concurrent games
            output_dir: Directory to save progress log
        """
        self.total_games = total_games
        self.max_concurrent = max_concurrent
        self.output_dir = output_dir
        
        # Game state tracking
        self.waiting_games = set(range(total_games))
        self.running_games = {}  # game_id -> start_time
        self.completed_games = {}  # game_id -> result
        self.failed_games = {}  # game_id -> error
        
        # Progress log file
        self.log_file = output_dir / 'tournament_progress.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log
        with open(self.log_file, 'w') as f:
            f.write(f"Tournament Progress Log\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total games: {total_games}\n")
            f.write(f"Max concurrent: {max_concurrent}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")
    
    def start_game(self, game_id: str):
        """Mark a game as started.
        
        Args:
            game_id: ID of the game
        """
        start_time = datetime.now()
        self.running_games[game_id] = start_time
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"[{start_time.strftime('%H:%M:%S')}] 🎮 STARTED: {game_id}\n")
        
        # Print status update
        self._print_status(f"🎮 STARTED: {game_id}")
    
    def complete_game(self, game_id: str, success: bool, winner: str = None, error: str = None):
        """Mark a game as completed.
        
        Args:
            game_id: ID of the game
            success: Whether game completed successfully
            winner: Winner of the game (if successful)
            error: Error message (if failed)
        """
        end_time = datetime.now()
        start_time = self.running_games.pop(game_id, end_time)
        duration = (end_time - start_time).total_seconds()
        
        if success:
            self.completed_games[game_id] = {
                'winner': winner,
                'duration': duration,
                'end_time': end_time
            }
            status = f"✅ COMPLETED: {game_id} - Winner: {winner} ({duration:.1f}s)"
        else:
            self.failed_games[game_id] = {
                'error': error,
                'duration': duration,
                'end_time': end_time
            }
            status = f"❌ FAILED: {game_id} - Error: {error[:50]}... ({duration:.1f}s)"
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"[{end_time.strftime('%H:%M:%S')}] {status}\n")
        
        # Print status update
        self._print_status(status)
    
    def _print_status(self, event: str):
        """Print current status with game counts.
        
        Args:
            event: Event message to display
        """
        completed = len(self.completed_games)
        failed = len(self.failed_games)
        running = len(self.running_games)
        waiting = self.total_games - completed - failed - running
        
        print(f"\n{event}")
        print(f"  📊 Progress: {completed + failed}/{self.total_games} complete "
              f"({completed} ✅, {failed} ❌) | "
              f"{running} 🎮 running | {waiting} ⏳ waiting")
        
        # Show which games are currently running
        if self.running_games:
            running_list = list(self.running_games.keys())[:5]  # Show first 5
            running_str = ", ".join(running_list)
            if len(self.running_games) > 5:
                running_str += f" ... (+{len(self.running_games) - 5} more)"
            print(f"  🎮 Running: {running_str}")
        
        # Log detailed state to file periodically
        if (completed + failed) % 5 == 0 or completed + failed == self.total_games:
            self._log_detailed_state()
    
    def _log_detailed_state(self):
        """Log detailed state to file."""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'─'*80}\n")
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Status Update\n")
            f.write(f"  Completed: {len(self.completed_games)}\n")
            f.write(f"  Failed: {len(self.failed_games)}\n")
            f.write(f"  Running: {len(self.running_games)}\n")
            f.write(f"  Waiting: {self.total_games - len(self.completed_games) - len(self.failed_games) - len(self.running_games)}\n")
            
            if self.running_games:
                f.write(f"\n  Currently running:\n")
                for game_id, start_time in self.running_games.items():
                    elapsed = (datetime.now() - start_time).total_seconds()
                    f.write(f"    - {game_id} (running for {elapsed:.1f}s)\n")
            
            f.write(f"{'─'*80}\n\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.
        
        Returns:
            Summary dictionary
        """
        return {
            'total_games': self.total_games,
            'completed': len(self.completed_games),
            'failed': len(self.failed_games),
            'success_rate': len(self.completed_games) / self.total_games if self.total_games > 0 else 0,
        }


class TournamentRunner:
    """Generic tournament runner for any social deduction game."""
    
    def __init__(self, config_path: Path):
        """Initialize tournament runner.
        
        Args:
            config_path: Path to tournament configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load tournament configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required = ['name', 'game', 'game_config', 'schedule_file', 'schedule_format']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        # Set defaults
        config.setdefault('output_dir', f"experiments/tournaments/{config['game']}")
        config.setdefault('log_games', True)
        config.setdefault('log_private_info', True)
        config.setdefault('max_concurrent_games', 3)
        config.setdefault('timeout_per_game', 3600)
        config.setdefault('model_mapping', {})
        config.setdefault('agent_config', {})
        config.setdefault('metrics', [])
        
        return config
    
    def normalize_model_name(self, model_name: str) -> str:
        """Normalize model name using configured mappings.
        
        Args:
            model_name: Model name (can be short or full)
            
        Returns:
            Full model name for OpenRouter
        """
        model_name = model_name.strip()
        
        # If already in full format, return as is
        if "/" in model_name:
            return model_name
        
        # Look up in mapping
        mapping = self.config.get('model_mapping', {})
        if model_name in mapping:
            return mapping[model_name]
        
        # Fallback: return as is
        print(f"⚠️  Warning: Unknown model '{model_name}', using as-is")
        return model_name
    
    def parse_schedule(self) -> List[Dict[str, Any]]:
        """Parse tournament schedule from CSV file.
        
        Returns:
            List of game configurations
        """
        schedule_path = Path(self.config['schedule_file'])
        if not schedule_path.exists():
            raise FileNotFoundError(f"Schedule file not found: {schedule_path}")
        
        schedule_format = self.config['schedule_format']
        
        if schedule_format == 'werewolf':
            return self._parse_werewolf_schedule(schedule_path)
        elif schedule_format == 'amongus' or schedule_format == 'among_us':
            return self._parse_amongus_schedule(schedule_path)
        elif schedule_format == 'avalon':
            return self._parse_avalon_schedule(schedule_path)
        elif schedule_format == 'spyfall':
            return self._parse_spyfall_schedule(schedule_path)
        elif schedule_format == 'sheriff':
            return self._parse_sheriff_schedule(schedule_path)
        elif schedule_format == 'secret_hitler':
            return self._parse_secret_hitler_schedule(schedule_path)
        else:
            raise ValueError(f"Unknown schedule format: {schedule_format}")
    
    def _parse_werewolf_schedule(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Parse werewolf schedule: game_id,villager_model,werewolf_model"""
        games = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                villager_model = self.normalize_model_name(row['villager_model'])
                werewolf_model = self.normalize_model_name(row['werewolf_model'])
                
                # Get player counts from config
                n_villagers = self.config['game_config'].get('n_players', 6) - \
                             self.config['game_config'].get('n_werewolves', 2)
                n_werewolves = self.config['game_config'].get('n_werewolves', 2)
                
                games.append({
                    'game_id': row['game_id'],
                    'players': [villager_model] * n_villagers + [werewolf_model] * n_werewolves,
                    'role_assignment': {
                        'villagers': list(range(n_villagers)),
                        'werewolves': list(range(n_villagers, n_villagers + n_werewolves)),
                    }
                })
        return games
    
    def _parse_amongus_schedule(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Parse Among Us schedule: game_id,impostors,crewmates"""
        games = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                impostors = [self.normalize_model_name(m.strip()) 
                           for m in row['impostors'].split(',')]
                crewmates = [self.normalize_model_name(m.strip()) 
                           for m in row['crewmates'].split(',')]
                
                games.append({
                    'game_id': row['game_id'],
                    'players': impostors + crewmates,
                    'role_assignment': {
                        'impostors': list(range(len(impostors))),
                        'crewmates': list(range(len(impostors), len(impostors) + len(crewmates))),
                    }
                })
        return games
    
    def _parse_avalon_schedule(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Parse Avalon schedule: game_id,players,good,evil"""
        games = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                good = [self.normalize_model_name(m.strip()) 
                       for m in row['good'].split(';')]
                evil = [self.normalize_model_name(m.strip()) 
                       for m in row['evil'].split(';')]
                
                games.append({
                    'game_id': row['game_id'],
                    'players': good + evil,
                    'role_assignment': {
                        'good': list(range(len(good))),
                        'evil': list(range(len(good), len(good) + len(evil))),
                    }
                })
        return games
    
    def _parse_spyfall_schedule(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Parse Spyfall schedule: game_id,players,spy"""
        games = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                players = [self.normalize_model_name(m.strip()) 
                          for m in row['players'].split(';')]
                spy_model = self.normalize_model_name(row['spy'].strip())
                
                # Find which player is the spy
                spy_idx = None
                for i, player in enumerate(players):
                    if player == spy_model:
                        spy_idx = i
                        break
                
                # If spy is not in players list, pick random position and replace
                if spy_idx is None:
                    import random
                    spy_idx = random.randrange(len(players))
                    players[spy_idx] = spy_model
                
                games.append({
                    'game_id': row['game_id'],
                    'players': players,
                    'role_assignment': {
                        'spy': spy_idx,
                    }
                })
        return games
    
    def _parse_sheriff_schedule(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Parse Sheriff schedule: game_id,player1,player2,player3,player4"""
        games = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                players = [
                    self.normalize_model_name(row['player1'].strip()),
                    self.normalize_model_name(row['player2'].strip()),
                    self.normalize_model_name(row['player3'].strip()),
                    self.normalize_model_name(row['player4'].strip()),
                ]
                
                games.append({
                    'game_id': row['game_id'],
                    'players': players,
                    'role_assignment': {}  # Sheriff rotates, no fixed roles
                })
        return games
    
    def _parse_secret_hitler_schedule(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Parse Secret Hitler schedule: game_id,liberals,fascists"""
        games = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                liberals = [self.normalize_model_name(m.strip()) 
                           for m in row['liberals'].split(',')]
                fascists = [self.normalize_model_name(m.strip()) 
                           for m in row['fascists'].split(',')]
                
                games.append({
                    'game_id': row['game_id'],
                    'players': liberals + fascists,
                    'role_assignment': {
                        'liberals': list(range(len(liberals))),
                        'fascists': list(range(len(liberals), len(liberals) + len(fascists))),
                    }
                })
        return games
    
    def create_agents(self, players: List[str]) -> List[OpenRouterAgent]:
        """Create agents from player model names.
        
        Args:
            players: List of model names
            
        Returns:
            List of configured agents
        """
        agents = []
        agent_config = self.config.get('agent_config', {})
        
        for i, model in enumerate(players):
            agent = OpenRouterAgent(
                player_id=i,
                name=f"Agent_{i}",
                model=model,
                temperature=agent_config.get('temperature', 0.7),
                max_tokens=agent_config.get('max_tokens', 4096),
            )
            agents.append(agent)
        
        return agents
    
    def load_game_environment(self):
        """Load game environment and config classes dynamically.
        
        Returns:
            Tuple of (EnvClass, ConfigClass)
        """
        game = self.config['game']
        
        # Map game names to module paths
        game_module_map = {
            'werewolf': 'sdb.environments.werewolf',
            'amongus': 'sdb.environments.among_us',
            'among_us': 'sdb.environments.among_us',
            'avalon': 'sdb.environments.avalon',
            'spyfall': 'sdb.environments.spyfall',
            'sheriff': 'sdb.environments.sheriff',
            'secret_hitler': 'sdb.environments.secret_hitler',
        }
        
        if game not in game_module_map:
            raise ValueError(f"Unknown game type: {game}")
        
        module_path = game_module_map[game]
        
        # Import environment module
        env_module = importlib.import_module(f"{module_path}.env")
        config_module = importlib.import_module(f"{module_path}.config")
        
        # Get class names (capitalize first letter + Env/Config)
        game_name_parts = game.replace('_', ' ').title().replace(' ', '')
        env_class_name = f"{game_name_parts}Env"
        config_class_name = f"{game_name_parts}Config"
        
        env_class = getattr(env_module, env_class_name)
        config_class = getattr(config_module, config_class_name)
        
        return env_class, config_class
    
    async def run_game(
        self,
        game_config: Dict[str, Any],
        output_dir: Path,
        progress_tracker: 'TournamentProgressTracker'
    ) -> Dict[str, Any]:
        """Run a single game.
        
        Args:
            game_config: Game configuration from schedule
            output_dir: Output directory for logs
            progress_tracker: Progress tracking object
            
        Returns:
            Game result dictionary
        """
        game_id = game_config['game_id']
        game_start = datetime.now()
        
        try:
            # Mark game as started
            progress_tracker.start_game(game_id)
            
            # Create agents
            agents = self.create_agents(game_config['players'])
            
            # Load environment classes
            EnvClass, ConfigClass = self.load_game_environment()
            
            # Create game config
            game_config_params = self.config['game_config'].copy()
            game_config_params['n_players'] = len(game_config['players'])
            
            env_config = ConfigClass(**game_config_params)
            
            # Setup logger
            if self.config['log_games']:
                log_dir = output_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                logger = GameLogger(
                    game_id=game_config['game_id'],
                    output_dir=log_dir,
                    log_private=self.config.get('log_private_info', True)
                )
            else:
                logger = None
            
            # Create environment with agents
            # Pass role_assignment if provided (for games that support it)
            env_kwargs = {
                'agents': agents,
                'config': env_config,
                'game_id': game_config['game_id'],
                'logger': logger
            }
            if 'role_assignment' in game_config:
                env_kwargs['role_assignment'] = game_config['role_assignment']
            
            env = EnvClass(**env_kwargs)
            
            # Run game (check if async or sync)
            if hasattr(env, 'play_game') and asyncio.iscoroutinefunction(env.play_game):
                result = await env.play_game()
            elif hasattr(env, 'play_game_async'):
                result = await env.play_game_async()
            else:
                result = env.play_game()
            
            # Extract result from GameResult object
            game_result = {
                'game_id': game_config['game_id'],
                'winner': result.winner,
                'win_reason': result.win_reason,
                'rounds': result.num_rounds,
                'duration_seconds': result.duration_seconds,
                'players': game_config['players'],
                'role_assignment': game_config.get('role_assignment', {}),
                'player_stats': result.player_stats,
            }
            
            # Add game-specific metrics from metadata
            if result.metadata:
                for key in ['task_completion', 'missions', 'policies_enacted', 'final_score']:
                    if key in result.metadata:
                        game_result[key] = result.metadata[key]
            
            # Mark game as completed
            progress_tracker.complete_game(game_id, success=True, winner=result.winner)
            
            return game_result
            
        except Exception as e:
            # Mark game as failed
            progress_tracker.complete_game(game_id, success=False, error=str(e))
            
            return {
                'game_id': game_config['game_id'],
                'error': str(e),
                'traceback': traceback.format_exc(),
                'duration_seconds': (datetime.now() - game_start).total_seconds(),
            }
    
    async def run_tournament(self) -> Dict[str, Any]:
        """Run the full tournament.
        
        Returns:
            Tournament results dictionary
        """
        print(f"\n{'='*80}")
        print(f"🏆 TOURNAMENT: {self.config['name']}")
        print(f"{'='*80}")
        print(f"🎮 Game: {self.config['game']}")
        print(f"📋 Schedule: {self.config['schedule_file']}")
        print(f"📁 Output: {self.config['output_dir']}")
        print(f"⚡ Max concurrent: {self.config['max_concurrent_games']}")
        print(f"{'='*80}\n")
        
        # Parse schedule
        games = self.parse_schedule()
        print(f"✅ Loaded {len(games)} games from schedule\n")
        
        # Create output directory
        output_dir = Path(self.config['output_dir'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = output_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_save_path = output_dir / 'tournament_config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Setup progress tracking
        progress_tracker = TournamentProgressTracker(
            total_games=len(games),
            max_concurrent=self.config['max_concurrent_games'],
            output_dir=output_dir
        )
        
        # Run games with concurrency control
        start_time = datetime.now()
        semaphore = asyncio.Semaphore(self.config['max_concurrent_games'])
        
        async def run_game_with_semaphore(game_config):
            async with semaphore:
                return await self.run_game(game_config, output_dir, progress_tracker)
        
        # Run all games
        print(f"🎮 Running {len(games)} games (max {self.config['max_concurrent_games']} concurrent)...\n")
        results = await asyncio.gather(*[run_game_with_semaphore(game) for game in games])
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Count successes and failures
        successes = sum(1 for r in results if 'error' not in r)
        failures = len(results) - successes
        
        # Get progress summary
        progress_summary = progress_tracker.get_summary()
        
        # Save results
        tournament_results = {
            'tournament_id': f"{self.config['game']}_{timestamp}",
            'name': self.config['name'],
            'game_type': self.config['game'],
            'config': self.config,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_games': len(games),
            'successful_games': successes,
            'failed_games': failures,
            'games': results,
        }
        
        results_file = output_dir / 'tournament_results.json'
        with open(results_file, 'w') as f:
            json.dump(tournament_results, f, indent=2, cls=TournamentJSONEncoder)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"🏁 TOURNAMENT COMPLETE")
        print(f"{'='*80}")
        print(f"⏱️  Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        print(f"✅ Successful: {successes}/{len(games)}")
        print(f"❌ Failed: {failures}/{len(games)}")
        print(f"📊 Success rate: {progress_summary['success_rate']*100:.1f}%")
        print(f"📊 Results saved to: {results_file}")
        print(f"📝 Game logs saved to: {output_dir / 'logs'}")
        print(f"📈 Progress log: {progress_tracker.log_file}")
        print(f"📄 Config saved to: {config_save_path}")
        print(f"{'='*80}\n")
        
        return tournament_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tournaments from configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Werewolf tournament
  python scripts/run_tournament.py configs/tournaments/werewolf_tournament.yaml

  # Run Among Us tournament
  python scripts/run_tournament.py configs/tournaments/amongus_tournament.yaml

  # Run Avalon tournament
  python scripts/run_tournament.py configs/tournaments/avalon_tournament.yaml
        """
    )
    
    parser.add_argument(
        'config',
        type=Path,
        help='Path to tournament configuration YAML file'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        help='Override maximum number of concurrent games'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Override output directory'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("⚠️  Warning: OPENROUTER_API_KEY environment variable not set")
        print("   Set it with: export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Create runner
    try:
        runner = TournamentRunner(args.config)
        
        # Apply overrides
        if args.max_concurrent:
            runner.config['max_concurrent_games'] = args.max_concurrent
        if args.output_dir:
            runner.config['output_dir'] = str(args.output_dir)
        
        # Run tournament
        asyncio.run(runner.run_tournament())
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tournament interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Tournament failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
