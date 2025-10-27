"""Evaluator for Werewolf tournaments."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from evaluations.base_evaluator import BaseEvaluator, GameResult, read_jsonl, extract_model_name


class WerewolfEvaluator(BaseEvaluator):
    """Evaluator for Werewolf game tournaments."""
    
    def parse_game_log(self, log_file: Path) -> Optional[GameResult]:
        """Parse Werewolf game log.
        
        Args:
            log_file: Path to .jsonl log file
            
        Returns:
            GameResult or None
        """
        entries = read_jsonl(log_file)
        if not entries:
            return None
        
        # Extract game info
        game_start = next((e for e in entries if e.get('event_type') == 'GAME_START'), None)
        game_end = next((e for e in entries if e.get('event_type') == 'GAME_END'), None)
        
        if not game_start or not game_end:
            return None
        
        game_id = game_start.get('game_id', log_file.stem)
        
        # Get player-role mapping
        player_roles = {}
        agents_info = next((e for e in entries 
                           if e.get('data', {}).get('action') == 'agent_metadata' 
                           or e.get('data', {}).get('event') == 'agent_metadata'), None)
        
        if not agents_info:
            return None
        
        agents = agents_info['data']['agents']
        
        # Get role assignments (they're in a PLAYER_ACTION event with action: role_assignment)
        role_assignment = next((e for e in entries 
                               if e.get('data', {}).get('action') == 'role_assignment'), None)
        
        # Store as player_id -> (model, role) to handle duplicate models
        player_id_mapping = {}
        
        if role_assignment:
            roles = role_assignment['data'].get('roles', [])
            for pid, agent_data in agents.items():
                model = extract_model_name(agent_data['model'])
                role_idx = int(pid)
                if role_idx < len(roles):
                    role = roles[role_idx]
                else:
                    role = 'unknown'
                player_id_mapping[int(pid)] = (model, role)
                # Use player_id as unique key to avoid overwriting
                player_roles[f"{model}_p{pid}"] = role
        else:
            # Fallback: look for individual ROLE_ASSIGNMENT events
            role_assignments = [e for e in entries if e.get('event_type') == 'ROLE_ASSIGNMENT']
            for pid, agent_data in agents.items():
                model = extract_model_name(agent_data['model'])
                player_id_mapping[int(pid)] = (model, None)
                player_roles[f"{model}_p{pid}"] = None
            
            for assignment in role_assignments:
                player_id = str(assignment.get('player_id'))
                role = assignment['data'].get('role')
                if player_id in agents:
                    model = extract_model_name(agents[player_id]['model'])
                    player_id_mapping[int(player_id)] = (model, role)
                    player_roles[f"{model}_p{player_id}"] = role
        
        # Get winner
        winner_data = game_end.get('data', {})
        winner = winner_data.get('winner', 'unknown')
        win_reason = winner_data.get('reason', '')
        
        # Count rounds
        rounds = max((e.get('round_number', 0) for e in entries), default=0) + 1
        
        # Calculate duration
        if entries:
            start_time = pd.to_datetime(entries[0]['timestamp'])
            end_time = pd.to_datetime(entries[-1]['timestamp'])
            duration = (end_time - start_time).total_seconds()
        else:
            duration = 0
        
        # Player stats (use player_id_mapping for accurate tracking)
        player_stats = {}
        for pid, (model, role) in player_id_mapping.items():
            survived, death_reason = self._check_death_by_id(pid, entries)
            player_key = f"{model}_p{pid}"
            player_stats[player_key] = {
                'role': role,
                'model': model,  # Store actual model name
                'survived': survived,
                'death_reason': death_reason,  # 'lynch', 'night_kill', or None
            }
        
        return GameResult(
            game_id=game_id,
            winner=winner,
            win_reason=win_reason,
            num_rounds=rounds,
            duration=duration,
            players=list(player_roles.keys()),
            player_roles=player_roles,
            player_stats=player_stats
        )
    
    def _check_death_by_id(self, player_id: int, entries: List[Dict]) -> tuple:
        """Check if player died and how by player_id.
        
        Args:
            player_id: Player ID to check
            entries: Log entries
        
        Returns:
            (survived: bool, death_reason: str or None)
            death_reason can be 'lynch', 'night_kill', or None
        """
        # Check for death events
        death_events = [e for e in entries if e.get('event_type') == 'PLAYER_ELIMINATED']
        for event in death_events:
            # Check both event.player_id and data.player_id
            eliminated_id = event.get('player_id')
            if eliminated_id is None:
                eliminated_id = event.get('data', {}).get('player_id')
            
            if eliminated_id == player_id:
                # Determine death reason from 'by' field
                death_by = event.get('data', {}).get('by', '')
                if death_by == 'werewolves':
                    return False, 'night_kill'
                elif death_by == 'lynch' or death_by == 'vote':
                    return False, 'lynch'
                else:
                    # Default to night kill if not specified
                    return False, 'night_kill'
        
        return True, None
    
    def _is_winner(self, player: str, result: GameResult) -> bool:
        """Check if player won (team-based)."""
        role = result.player_roles.get(player)
        if not role:
            return False
        
        # Werewolves/wolves win or villagers/village win
        winner = result.winner.lower() if result.winner else ''
        if winner.startswith('werewolv') or winner.startswith('wolf') or winner.startswith('wolv'):
            return role == 'werewolf'
        elif winner.startswith('village') or winner.startswith('villager') or winner.startswith('town'):
            return role in ['villager', 'seer', 'doctor']
        
        return False
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate werewolf-specific summary table with detailed columns."""
        if not self.results:
            raise ValueError("No results loaded. Call load_all_games() first.")
        
        from collections import defaultdict
        
        # Collect detailed stats per model
        model_stats = defaultdict(lambda: {
            'games': 0,
            'werewolf_games': 0,
            'werewolf_wins': 0,
            'villager_games': 0,
            'villager_wins': 0,
            'seer_games': 0,
            'seer_wins': 0,
            'doctor_games': 0,
            'doctor_wins': 0,
            'lynched': 0,
            'night_killed': 0,
        })
        
        for result in self.results:
            for player_key, role in result.player_roles.items():
                if not role:
                    continue
                
                # Extract actual model name from player_stats
                model = result.player_stats.get(player_key, {}).get('model', player_key.split('_p')[0])
                
                stats = model_stats[model]
                stats['games'] += 1
                
                # Count games by role
                if role == 'werewolf':
                    stats['werewolf_games'] += 1
                    if self._is_winner(player_key, result):
                        stats['werewolf_wins'] += 1
                elif role == 'villager':
                    stats['villager_games'] += 1
                    if self._is_winner(player_key, result):
                        stats['villager_wins'] += 1
                elif role == 'seer':
                    stats['seer_games'] += 1
                    if self._is_winner(player_key, result):
                        stats['seer_wins'] += 1
                elif role == 'doctor':
                    stats['doctor_games'] += 1
                    if self._is_winner(player_key, result):
                        stats['doctor_wins'] += 1
                
                # Check if lynched or night killed
                if not result.player_stats.get(player_key, {}).get('survived', True):
                    # Need to check how they died
                    stats['night_killed'] += self._count_night_kills(player_key, result)
                    stats['lynched'] += self._count_lynches(player_key, result)
        
        # Build dataframe
        rows = []
        for model in sorted(model_stats.keys()):
            stats = model_stats[model]
            
            # Calculate win rates
            wolf_wr = (stats['werewolf_wins'] / stats['werewolf_games'] * 100) if stats['werewolf_games'] > 0 else 0
            
            # Town WR = (villager + seer + doctor wins) / (villager + seer + doctor games)
            town_games = stats['villager_games'] + stats['seer_games'] + stats['doctor_games']
            town_wins = stats['villager_wins'] + stats['seer_wins'] + stats['doctor_wins']
            town_wr = (town_wins / town_games * 100) if town_games > 0 else 0
            
            rows.append({
                'Model': model,
                'Games': stats['games'],
                'Wolf WR': f"{wolf_wr:.1f}%",
                'Town WR': f"{town_wr:.1f}%",
                'Lynch': stats['lynched'],
                'Seer': stats['seer_games'],
                'Doctor': stats['doctor_games'],
                'NK': stats['night_killed'],
            })
        
        return pd.DataFrame(rows)
    
    def _count_night_kills(self, player: str, result: GameResult) -> int:
        """Count if player was night killed in this game."""
        death_reason = result.player_stats.get(player, {}).get('death_reason')
        return 1 if death_reason == 'night_kill' else 0
    
    def _count_lynches(self, player: str, result: GameResult) -> int:
        """Count if player was lynched in this game."""
        death_reason = result.player_stats.get(player, {}).get('death_reason')
        return 1 if death_reason == 'lynch' else 0
    
    def generate_detailed_stats(self) -> Dict[str, Any]:
        """Generate Werewolf-specific statistics."""
        role_stats = defaultdict(lambda: defaultdict(int))
        
        for result in self.results:
            for player, role in result.player_roles.items():
                if not role:
                    continue
                    
                role_stats[player][f'{role}_games'] += 1
                
                if self._is_winner(player, result):
                    role_stats[player][f'{role}_wins'] += 1
                
                if result.player_stats.get(player, {}).get('survived'):
                    role_stats[player][f'{role}_survived'] += 1
        
        # Calculate win rates by role
        detailed = {}
        for player, stats in role_stats.items():
            detailed[player] = dict(stats)
            
            # Calculate role-specific win rates
            for role in ['werewolf', 'villager', 'seer', 'doctor']:
                games = stats.get(f'{role}_games', 0)
                if games > 0:
                    wins = stats.get(f'{role}_wins', 0)
                    detailed[player][f'{role}_win_rate'] = wins / games
        
        return detailed


def main():
    """Run Werewolf evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Werewolf tournament')
    parser.add_argument('tournament_dir', type=Path, help='Tournament directory')
    parser.add_argument('--output', type=Path, help='Output directory (default: tournament_dir)')
    args = parser.parse_args()
    
    evaluator = WerewolfEvaluator(args.tournament_dir)
    evaluator.load_all_games()
    evaluator.print_summary()
    evaluator.save_tables(args.output)


if __name__ == '__main__':
    main()

