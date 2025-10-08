"""Round-robin tournament implementation."""

import asyncio
import itertools
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from sdb.core.base_agent import BaseAgent
from sdb.tournament.base import BaseTournament, TournamentConfig, TournamentResult, GameRecord
from sdb.environments.registry import get_env
from sdb.core.utils import generate_game_id


class RoundRobinTournament(BaseTournament):
    """Round-robin tournament where every combination of players competes.
    
    Suitable for comparing different agents or agent configurations.
    """
    
    def __init__(self, config: TournamentConfig):
        """Initialize round-robin tournament.
        
        Args:
            config: Tournament configuration
        """
        super().__init__(config)
    
    async def run(self, agents: List[BaseAgent]) -> TournamentResult:
        """Run round-robin tournament.
        
        Args:
            agents: List of agents to compete
            
        Returns:
            Tournament results
        """
        self.start_time = datetime.now()
        tournament_id = f"roundrobin_{self.config.name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n🏆 Starting Round-Robin Tournament: {self.config.name}")
        print(f"   Environment: {self.config.environment}")
        print(f"   Agents: {len(agents)}")
        print(f"   Games: {self.config.num_games}")
        print(f"   Tournament ID: {tournament_id}\n")
        
        # Generate all matchups
        matchups = self._generate_matchups(agents)
        print(f"   Generated {len(matchups)} matchups")
        
        # Get environment class
        env_class = get_env(self.config.environment)
        
        # Run all games
        game_number = 0
        for matchup in matchups:
            game_number += 1
            print(f"\n📊 Game {game_number}/{len(matchups)}")
            print(f"   Players: {matchup}")
            
            # Select agents for this game
            game_agents = [agents[i] for i in matchup]
            
            # Reset agents
            for agent in game_agents:
                agent.reset()
            
            # Create environment
            # Note: This assumes environment can be instantiated with agents
            # Actual implementation depends on environment interface
            env = env_class(
                agents=game_agents,
                game_id=f"{tournament_id}_game_{game_number}",
                logger_enabled=self.config.log_games
            )
            
            # Run game
            start = datetime.now()
            try:
                result = await env.play_game()
                end = datetime.now()
                
                # Record game
                game_record = GameRecord(
                    game_id=result.get("game_id", f"game_{game_number}"),
                    game_number=game_number,
                    players=matchup,
                    winner=result.get("winner", "unknown"),
                    win_reason=result.get("reason", ""),
                    num_rounds=result.get("rounds", 0),
                    duration_seconds=(end - start).total_seconds(),
                    player_stats=result.get("player_stats", {}),
                    metadata=result
                )
                self.games.append(game_record)
                
                print(f"   ✅ Winner: {game_record.winner}")
                print(f"   Duration: {game_record.duration_seconds:.1f}s")
                
            except Exception as e:
                print(f"   ❌ Game failed: {e}")
                import traceback
                traceback.print_exc()
        
        self.end_time = datetime.now()
        
        # Calculate rankings and stats
        rankings = self._calculate_rankings(agents)
        player_stats = self._aggregate_player_stats(agents)
        
        # Create result
        result = TournamentResult(
            tournament_id=tournament_id,
            config=self.config,
            start_time=self.start_time,
            end_time=self.end_time,
            games=self.games,
            player_stats=player_stats,
            rankings=rankings,
            metadata={"type": "round_robin"}
        )
        
        # Save results
        output_file = self.config.output_dir / f"{tournament_id}.json"
        result.save(output_file)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _generate_matchups(self, agents: List[BaseAgent]) -> List[List[int]]:
        """Generate all possible combinations of players for games.
        
        Args:
            agents: List of agents
            
        Returns:
            List of matchups
        """
        # Get environment requirements
        env_class = get_env(self.config.environment)
        # Assume we want to use all agents in each game
        # For games with variable player counts, this would need configuration
        
        # For now, assume all agents play in each game
        # and we repeat num_games times
        agent_ids = [agent.player_id for agent in agents]
        matchups = [agent_ids for _ in range(self.config.num_games)]
        
        return matchups
    
    def _print_summary(self, result: TournamentResult) -> None:
        """Print tournament summary.
        
        Args:
            result: Tournament results
        """
        print("\n" + "="*80)
        print("🏆 TOURNAMENT RESULTS")
        print("="*80)
        print(f"Tournament: {result.config.name}")
        print(f"Duration: {(result.end_time - result.start_time).total_seconds():.1f}s")
        print(f"Games Played: {len(result.games)}")
        
        print("\n📊 Rankings:")
        for rank, (player_id, score) in enumerate(result.rankings, 1):
            stats = result.player_stats[player_id]
            print(f"   {rank}. Player {player_id}: {score} points "
                  f"({stats['wins']}/{stats['games_played']} wins, "
                  f"{stats['win_rate']:.1%} win rate)")
        
        print("\n" + "="*80)

