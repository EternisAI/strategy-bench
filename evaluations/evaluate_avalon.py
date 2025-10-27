"""Evaluator for Avalon tournaments."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from evaluations.base_evaluator import BaseEvaluator, GameResult, read_jsonl, extract_model_name


class AvalonEvaluator(BaseEvaluator):
    """Evaluator for Avalon game tournaments."""
    
    def parse_game_log(self, log_file: Path) -> Optional[GameResult]:
        """Parse Avalon game log.
        
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
        
        if not game_start:
            return None
        
        game_id = game_start.get('game_id', log_file.stem)
        
        # Get player-role mapping
        player_roles = {}
        player_id_mapping = {}  # pid -> (model, role, team)
        
        # Find agent metadata
        agents_info = next((e for e in entries 
                           if e.get('event_type') == 'GAME_START' and 'agents' in e.get('data', {})), None)
        
        # Find role assignments (INFO events with event: role_assignment)
        role_assignments = [e for e in entries 
                           if e.get('event_type') == 'INFO' 
                           and e.get('data', {}).get('event') == 'role_assignment']
        
        if not agents_info or not role_assignments:
            return None
        
        agents = agents_info['data']['agents']
        
        # Map roles by player ID
        for assignment in role_assignments:
            player_id = assignment.get('player_id')
            data = assignment.get('data', {})
            role = data.get('role')
            team = data.get('team')
            
            if player_id is not None and str(player_id) in agents:
                model = extract_model_name(agents[str(player_id)]['model'])
                player_key = f"{model}_p{player_id}"
                player_roles[player_key] = role
                player_id_mapping[player_id] = (model, role, team)
        
        # Count quests
        quest_results = [e for e in entries if e.get('event_type') == 'QUEST_RESULT']
        num_quests = len(quest_results)
        
        # Get winner
        if game_end:
            winner_data = game_end.get('data', {})
            winner = winner_data.get('winner', 'unknown')
            win_reason = winner_data.get('reason', '')
        else:
            # Infer winner from quest results if game is incomplete
            successes = sum(1 for q in quest_results if q.get('data', {}).get('succeeded', False))
            failures = sum(1 for q in quest_results if not q.get('data', {}).get('succeeded', True))
            
            if successes >= 3:
                winner = 'good'
                win_reason = '3 quests succeeded'
            elif failures >= 3:
                winner = 'evil'
                win_reason = '3 quests failed'
            else:
                winner = 'incomplete'
                win_reason = 'Game not finished'
        
        # Count rounds
        rounds = max((e.get('round_number', 0) for e in entries), default=0) + 1
        
        # Calculate duration
        if entries:
            start_time = pd.to_datetime(entries[0]['timestamp'])
            end_time = pd.to_datetime(entries[-1]['timestamp'])
            duration = (end_time - start_time).total_seconds()
        else:
            duration = 0
        
        # Player stats
        player_stats = {}
        for player_key, role in player_roles.items():
            player_stats[player_key] = {
                'role': role,
                'team': player_id_mapping.get(int(player_key.split('_p')[-1]), (None, None, None))[2],
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
    
    def _count_quests_on(self, model: str, quest_results: List[Dict], agents_info: Dict) -> int:
        """Count number of quests player participated in."""
        if not agents_info:
            return 0
        
        # Find player_id for this model
        player_id = None
        for pid, agent_data in agents_info['data']['agents'].items():
            if extract_model_name(agent_data['model']) == model:
                player_id = int(pid)
                break
        
        if player_id is None:
            return 0
        
        count = 0
        for quest in quest_results:
            team = quest.get('data', {}).get('team', [])
            if player_id in team:
                count += 1
        
        return count
    
    def _is_winner(self, player: str, result: GameResult) -> bool:
        """Check if player won (team-based)."""
        # Get team from player_stats
        team = result.player_stats.get(player, {}).get('team')
        if not team:
            return False
        
        # Match team with winner
        winner = result.winner.lower() if result.winner else ''
        
        if 'good' in winner:
            return team == 'good'
        elif 'evil' in winner or 'bad' in winner:
            return team == 'evil'
        
        return False
    
    def generate_summary_table(self, include_incomplete: bool = False) -> pd.DataFrame:
        """Generate Avalon-specific summary table with detailed columns.
        
        Args:
            include_incomplete: If False (default), only count completed games
        """
        if not self.results:
            raise ValueError("No results loaded. Call load_all_games() first.")
        
        # Filter results if needed
        results_to_analyze = self.results
        if not include_incomplete:
            results_to_analyze = [r for r in self.results if r.winner not in ['incomplete', 'unknown']]
        
        # Collect detailed stats per model
        model_stats = defaultdict(lambda: {
            'games': 0,
            'wins': 0,
            'good_games': 0,
            'good_wins': 0,
            'evil_games': 0,
            'evil_wins': 0,
            'evil_mission_wins': 0,  # Evil wins by 3 failed quests
            'evil_assassin_wins': 0,  # Evil wins by assassinating Merlin
        })
        
        for result in results_to_analyze:
            for player_key, role in result.player_roles.items():
                if not role:
                    continue
                
                # Extract model name (remove _pX suffix)
                model = '_'.join(player_key.split('_')[:-1]) if '_p' in player_key else player_key
                stats = model_stats[model]
                
                # Get team
                team = result.player_stats.get(player_key, {}).get('team')
                if not team:
                    continue
                
                # Count games
                stats['games'] += 1
                
                # Track team-based games
                if team == 'good':
                    stats['good_games'] += 1
                elif team == 'evil':
                    stats['evil_games'] += 1
                
                # Check if winner
                is_winner = self._is_winner(player_key, result)
                if is_winner:
                    stats['wins'] += 1
                    
                    if team == 'good':
                        stats['good_wins'] += 1
                    elif team == 'evil':
                        stats['evil_wins'] += 1
                        
                        # Determine how evil won
                        win_reason = result.win_reason.lower()
                        if 'assassin' in win_reason or 'merlin' in win_reason:
                            stats['evil_assassin_wins'] += 1
                        else:
                            # Default to mission win (3 failed quests)
                            stats['evil_mission_wins'] += 1
        
        # Build dataframe
        rows = []
        for model in sorted(model_stats.keys()):
            stats = model_stats[model]
            
            # Calculate metrics
            overall_wr = (stats['wins'] / stats['games'] * 100) if stats['games'] > 0 else 0
            good_wr = (stats['good_wins'] / stats['good_games'] * 100) if stats['good_games'] > 0 else 0
            evil_wr = (stats['evil_wins'] / stats['evil_games'] * 100) if stats['evil_games'] > 0 else 0
            
            rows.append({
                'Model': model,
                'Games': stats['games'],
                'Overall WR': f"{overall_wr:.1f}%",
                'Good Wins': stats['good_wins'],
                'Evil Wins': stats['evil_wins'],
                'Evil (Mission)': stats['evil_mission_wins'],
                'Evil (Assassin)': stats['evil_assassin_wins'],
                'Good WR': f"{good_wr:.1f}%",
                'Evil WR': f"{evil_wr:.1f}%",
            })
        
        return pd.DataFrame(rows)
    
    def print_summary(self):
        """Print summary to console with completion stats."""
        # Count completed vs incomplete
        completed = [r for r in self.results if r.winner not in ['incomplete', 'unknown']]
        incomplete = [r for r in self.results if r.winner in ['incomplete', 'unknown']]
        
        if incomplete:
            print(f"\n⚠️  Note: {len(incomplete)} of {len(self.results)} games incomplete ({len(incomplete)/len(self.results)*100:.1f}%)")
            print(f"    Showing results for {len(completed)} completed games only\n")
        
        df = self.generate_summary_table()
        print("\n" + "="*80)
        print("TOURNAMENT SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
    
    def generate_detailed_stats(self) -> Dict[str, Any]:
        """Generate Avalon-specific statistics."""
        role_stats = defaultdict(lambda: defaultdict(int))
        
        for result in self.results:
            for player, role in result.player_roles.items():
                if not role:
                    continue
                    
                role_stats[player][f'{role}_games'] += 1
                
                if self._is_winner(player, result):
                    role_stats[player][f'{role}_wins'] += 1
                
                # Quest participation
                quests = result.player_stats.get(player, {}).get('quests_completed', 0)
                role_stats[player][f'{role}_total_quests'] += quests
        
        # Calculate role-specific metrics
        detailed = {}
        for player, stats in role_stats.items():
            detailed[player] = dict(stats)
            
            # Calculate win rates by role
            for role in ['loyal_servant', 'merlin', 'percival', 'minion', 'assassin', 'morgana', 'mordred']:
                games = stats.get(f'{role}_games', 0)
                if games > 0:
                    wins = stats.get(f'{role}_wins', 0)
                    detailed[player][f'{role}_win_rate'] = wins / games
                    
                    # Average quests
                    total_quests = stats.get(f'{role}_total_quests', 0)
                    detailed[player][f'{role}_avg_quests'] = total_quests / games
        
        return detailed


def main():
    """Run Avalon evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Avalon tournament')
    parser.add_argument('tournament_dir', type=Path, help='Tournament directory')
    parser.add_argument('--output', type=Path, help='Output directory (default: tournament_dir)')
    args = parser.parse_args()
    
    evaluator = AvalonEvaluator(args.tournament_dir)
    evaluator.load_all_games()
    evaluator.print_summary()
    evaluator.save_tables(args.output)


if __name__ == '__main__':
    main()

