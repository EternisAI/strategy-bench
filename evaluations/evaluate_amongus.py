"""Evaluator for Among Us tournaments."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from evaluations.base_evaluator import BaseEvaluator, GameResult, read_jsonl, extract_model_name


class AmongUsEvaluator(BaseEvaluator):
    """Evaluator for Among Us game tournaments."""
    
    def parse_game_log(self, log_file: Path) -> Optional[GameResult]:
        """Parse Among Us game log.
        
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
        
        # Get role assignments
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
                    role = 'crewmate'
                player_id_mapping[int(pid)] = (model, role)
                player_roles[f"{model}_p{pid}"] = role
        else:
            # Fallback
            n_impostors = game_start['data'].get('n_impostors', 2)
            for pid, agent_data in agents.items():
                model = extract_model_name(agent_data['model'])
                pid_int = int(pid)
                role = 'impostor' if pid_int < n_impostors else 'crewmate'
                player_id_mapping[pid_int] = (model, role)
                player_roles[f"{model}_p{pid}"] = role
        
        # Get winner
        winner_data = game_end.get('data', {})
        winner = winner_data.get('winner', 'unknown')
        win_reason = winner_data.get('reason', '')
        
        # Get final task completion
        final_task_completion = winner_data.get('task_completion', 0.0)
        
        # Count rounds
        rounds = max((e.get('round_number', 0) for e in entries), default=0) + 1
        
        # Calculate duration
        if entries:
            start_time = pd.to_datetime(entries[0]['timestamp'])
            end_time = pd.to_datetime(entries[-1]['timestamp'])
            duration = (end_time - start_time).total_seconds()
        else:
            duration = 0
        
        # Get total rounds for survival calculation
        total_rounds = max((e.get('data', {}).get('round', 0) for e in entries 
                           if e.get('data', {}).get('round') is not None), default=0)
        
        # Player stats
        player_stats = {}
        for pid, (model, role) in player_id_mapping.items():
            survived, death_reason = self._check_death_by_id(pid, entries)
            kills = self._count_kills_by_id(pid, entries) if role == 'impostor' else 0
            tasks_done = self._count_tasks_by_id(pid, entries) if role == 'crewmate' else 0
            total_tasks = self._count_total_tasks(pid, entries) if role == 'crewmate' else 5
            
            # Calculate rounds alive
            death_round = self._get_death_round(pid, entries)
            rounds_alive = death_round if death_round else total_rounds
            
            # Calculate vote accuracy
            correct_votes, total_votes = self._calculate_vote_accuracy(pid, role, entries, player_id_mapping)
            
            player_key = f"{model}_p{pid}"
            player_stats[player_key] = {
                'role': role,
                'model': model,
                'survived': survived,
                'death_reason': death_reason,
                'kills': kills,
                'tasks_done': tasks_done,
                'total_tasks': total_tasks,
                'rounds_alive': rounds_alive,
                'total_rounds': total_rounds,
                'correct_votes': correct_votes,
                'total_votes': total_votes,
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
        """Check if player died and how."""
        death_events = [e for e in entries if e.get('event_type') == 'PLAYER_ELIMINATED']
        for event in death_events:
            eliminated_id = event.get('player_id')
            if eliminated_id is None:
                eliminated_id = event.get('data', {}).get('player_id')
            
            if eliminated_id == player_id:
                death_by = event.get('data', {}).get('by', '')
                if death_by == 'vote' or death_by == 'ejection':
                    return False, 'voted'
                elif death_by == 'kill' or death_by == 'impostor':
                    return False, 'killed'
                else:
                    return False, 'killed'
        
        return True, None
    
    def _count_kills_by_id(self, player_id: int, entries: List[Dict]) -> int:
        """Count kills performed by this impostor."""
        kills = 0
        # Look for PLAYER_ACTION events with action: kill
        for entry in entries:
            if entry.get('event_type') == 'PLAYER_ACTION':
                data = entry.get('data', {})
                if data.get('action') == 'kill' and data.get('killer') == player_id:
                    kills += 1
        return kills
    
    def _count_tasks_by_id(self, player_id: int, entries: List[Dict]) -> int:
        """Count tasks completed by this crewmate."""
        # Look for the final observation to see tasks done
        # Observations show "my_tasks": "X/Y" format
        for entry in reversed(entries):
            if entry.get('event_type') == 'INFO':
                obs = entry.get('data', {}).get('observation', {})
                if obs.get('round') is not None and entry.get('data', {}).get('player_id') == player_id:
                    my_tasks = obs.get('my_tasks', '0/5')
                    if isinstance(my_tasks, str) and '/' in my_tasks:
                        done, total = my_tasks.split('/')
                        return int(done)
        return 0
    
    def _count_total_tasks(self, player_id: int, entries: List[Dict]) -> int:
        """Get total tasks assigned to this crewmate."""
        # Look for initial observation to see total tasks
        for entry in entries:
            if entry.get('event_type') == 'INFO':
                obs = entry.get('data', {}).get('observation', {})
                if obs.get('round') == 0 and entry.get('data', {}).get('player_id') == player_id:
                    my_tasks = obs.get('my_tasks', '0/5')
                    if isinstance(my_tasks, str) and '/' in my_tasks:
                        done, total = my_tasks.split('/')
                        return int(total)
        return 5  # Default
    
    def _get_death_round(self, player_id: int, entries: List[Dict]) -> Optional[int]:
        """Get the round when this player died (None if survived)."""
        for entry in entries:
            if entry.get('event_type') == 'PLAYER_ELIMINATED':
                data = entry.get('data', {})
                if data.get('player_id') == player_id:
                    return data.get('round', 0)
        return None
    
    def _calculate_vote_accuracy(self, player_id: int, role: str, entries: List[Dict], 
                                  player_id_mapping: Dict[int, tuple]) -> tuple[int, int]:
        """Calculate vote accuracy for this player.
        
        For crewmates: correct = voting for impostor
        For impostors: correct = voting for crewmate
        
        Returns: (correct_votes, total_votes)
        """
        correct_votes = 0
        total_votes = 0
        
        # Find all ELECTION_RESULT events
        for entry in entries:
            if entry.get('event_type') == 'ELECTION_RESULT':
                votes = entry.get('data', {}).get('votes', {})
                
                # Check if this player voted
                if player_id in votes or str(player_id) in votes:
                    voted_for = votes.get(player_id) or votes.get(str(player_id))
                    
                    if voted_for is not None:
                        total_votes += 1
                        
                        # Check if vote was correct based on role
                        voted_for_role = player_id_mapping.get(voted_for, (None, None))[1]
                        
                        if role == 'crewmate' and voted_for_role == 'impostor':
                            correct_votes += 1
                        elif role == 'impostor' and voted_for_role == 'crewmate':
                            correct_votes += 1
        
        return correct_votes, total_votes
    
    def _is_winner(self, player: str, result: GameResult) -> bool:
        """Check if player won (team-based)."""
        role = result.player_roles.get(player)
        if not role:
            return False
        
        winner = result.winner.lower() if result.winner else ''
        if winner.startswith('impostor'):
            return role == 'impostor'
        elif winner.startswith('crewmate') or winner.startswith('crew'):
            return role == 'crewmate'
        
        return False
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate Among Us-specific summary table with detailed columns."""
        if not self.results:
            raise ValueError("No results loaded. Call load_all_games() first.")
        
        # Collect detailed stats per model
        model_stats = defaultdict(lambda: {
            'games': 0,
            'wins': 0,
            'impostor_games': 0,
            'impostor_wins': 0,
            'crewmate_games': 0,
            'crewmate_wins': 0,
            'crewmate_survived': 0,  # Games survived as crewmate
            'ejected': 0,
            'killed': 0,
            'survived': 0,
            'total_kills': 0,
            'total_tasks_done': 0,
            'total_tasks_possible': 0,
            'correct_votes': 0,
            'total_votes': 0,
        })
        
        for result in self.results:
            for player_key, role in result.player_roles.items():
                if not role:
                    continue
                
                model = result.player_stats.get(player_key, {}).get('model', player_key.split('_p')[0])
                stats = model_stats[model]
                stats['games'] += 1
                
                is_winner = self._is_winner(player_key, result)
                if is_winner:
                    stats['wins'] += 1
                
                # Count games by role
                if role == 'impostor':
                    stats['impostor_games'] += 1
                    if is_winner:
                        stats['impostor_wins'] += 1
                    # Track kills
                    stats['total_kills'] += result.player_stats.get(player_key, {}).get('kills', 0)
                elif role == 'crewmate':
                    stats['crewmate_games'] += 1
                    if is_winner:
                        stats['crewmate_wins'] += 1
                    # Track tasks
                    stats['total_tasks_done'] += result.player_stats.get(player_key, {}).get('tasks_done', 0)
                    stats['total_tasks_possible'] += result.player_stats.get(player_key, {}).get('total_tasks', 5)
                    
                    # Track crewmate survival (games survived as crewmate)
                    if result.player_stats.get(player_key, {}).get('survived', True):
                        stats['crewmate_survived'] += 1
                
                # Track survival and death
                if result.player_stats.get(player_key, {}).get('survived', True):
                    stats['survived'] += 1
                else:
                    death_reason = result.player_stats.get(player_key, {}).get('death_reason')
                    if death_reason == 'voted':
                        stats['ejected'] += 1
                    elif death_reason == 'killed':
                        stats['killed'] += 1
                
                # Track vote accuracy
                stats['correct_votes'] += result.player_stats.get(player_key, {}).get('correct_votes', 0)
                stats['total_votes'] += result.player_stats.get(player_key, {}).get('total_votes', 0)
        
        # Build dataframe with ranking
        rows = []
        for model in model_stats.keys():
            stats = model_stats[model]
            
            # Calculate metrics
            overall_wr = (stats['wins'] / stats['games'] * 100) if stats['games'] > 0 else 0
            crew_wr = (stats['crewmate_wins'] / stats['crewmate_games'] * 100) if stats['crewmate_games'] > 0 else 0
            imp_wr = (stats['impostor_wins'] / stats['impostor_games'] * 100) if stats['impostor_games'] > 0 else 0
            
            # Task completion rate (tasks done / tasks possible)
            task_per_game = (stats['total_tasks_done'] / stats['total_tasks_possible'] * 100) if stats['total_tasks_possible'] > 0 else 0
            
            # Vote accuracy: correct votes / total votes
            vote_accuracy = (stats['correct_votes'] / stats['total_votes'] * 100) if stats['total_votes'] > 0 else 0
            
            # Survival rate: games survived as crewmate / total crewmate games
            survival_pct = (stats['crewmate_survived'] / stats['crewmate_games'] * 100) if stats['crewmate_games'] > 0 else 0
            
            # Kills per impostor game
            kills_per_imp = (stats['total_kills'] / stats['impostor_games']) if stats['impostor_games'] > 0 else 0
            
            rows.append({
                'Model': model,
                'Games': stats['games'],
                'Overall WR': overall_wr,
                'CM WR': crew_wr,
                'IM WR': imp_wr,
                'Task/Game': task_per_game,
                'Vote %': vote_accuracy,  # Correct vote accuracy
                'Survival %': survival_pct,  # % of rounds alive
                'Kills/IM': kills_per_imp,
            })
        
        df = pd.DataFrame(rows)
        
        # Sort by Overall WR and add rank
        df = df.sort_values('Overall WR', ascending=False).reset_index(drop=True)
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        # Format percentages
        df['Overall WR'] = df['Overall WR'].apply(lambda x: f"{x:.1f}%")
        df['CM WR'] = df['CM WR'].apply(lambda x: f"{x:.1f}%")
        df['IM WR'] = df['IM WR'].apply(lambda x: f"{x:.1f}%")
        df['Task/Game'] = df['Task/Game'].apply(lambda x: f"{x:.0f}%")  # Task completion as %
        df['Vote %'] = df['Vote %'].apply(lambda x: f"{x:.0f}%")  # Vote survival as %
        df['Survival %'] = df['Survival %'].apply(lambda x: f"{x:.0f}%")
        df['Kills/IM'] = df['Kills/IM'].apply(lambda x: f"{x:.1f}")
        
        return df
    
    def generate_detailed_stats(self) -> Dict[str, Any]:
        """Generate Among Us-specific statistics."""
        role_stats = defaultdict(lambda: defaultdict(int))
        
        for result in self.results:
            for player_key, role in result.player_roles.items():
                if not role:
                    continue
                
                model = result.player_stats.get(player_key, {}).get('model', player_key.split('_p')[0])
                
                role_stats[model][f'{role}_games'] += 1
                
                if self._is_winner(player_key, result):
                    role_stats[model][f'{role}_wins'] += 1
        
        # Calculate role-specific metrics
        detailed = {}
        for player, stats in role_stats.items():
            detailed[player] = dict(stats)
            
            for role in ['impostor', 'crewmate']:
                games = stats.get(f'{role}_games', 0)
                if games > 0:
                    wins = stats.get(f'{role}_wins', 0)
                    detailed[player][f'{role}_win_rate'] = wins / games
        
        return detailed


def main():
    """Run Among Us evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Among Us tournament')
    parser.add_argument('tournament_dir', type=Path, help='Tournament directory')
    parser.add_argument('--output', type=Path, help='Output directory (default: tournament_dir)')
    args = parser.parse_args()
    
    evaluator = AmongUsEvaluator(args.tournament_dir)
    evaluator.load_all_games()
    evaluator.print_summary()
    evaluator.save_tables(args.output)


if __name__ == '__main__':
    main()
