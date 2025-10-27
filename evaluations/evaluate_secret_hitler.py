"""Evaluator for Secret Hitler tournaments."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from evaluations.base_evaluator import BaseEvaluator, GameResult, read_jsonl, extract_model_name


class SecretHitlerEvaluator(BaseEvaluator):
    """Evaluator for Secret Hitler game tournaments."""
    
    def parse_game_log(self, log_file: Path) -> Optional[GameResult]:
        """Parse Secret Hitler game log.
        
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
        player_id_mapping = {}  # pid -> (model, role)
        
        # Find agent metadata
        agents_info = next((e for e in entries 
                           if e.get('event_type') == 'GAME_START' and 'agents' in e.get('data', {})), None)
        
        # Find role assignments (PLAYER_ACTION with action: role_assignment)
        role_assignment = next((e for e in entries 
                               if e.get('event_type') == 'PLAYER_ACTION'
                               and e.get('data', {}).get('action') == 'role_assignment'), None)
        
        if not agents_info or not role_assignment:
            return None
        
        agents = agents_info['data']['agents']
        role_map = role_assignment['data'].get('role_map', {})
        
        # Map roles by player ID
        for pid_str, role in role_map.items():
            pid = int(pid_str)
            if pid_str in agents:
                model = extract_model_name(agents[pid_str]['model'])
                player_key = f"{model}_p{pid}"
                player_roles[player_key] = role.lower()  # LIBERAL -> liberal
                player_id_mapping[pid] = (model, role.lower())
        
        # Get winner
        if game_end:
            winner_data = game_end.get('data', {})
            winner = winner_data.get('winner', 'unknown')
            win_reason = winner_data.get('reason', '')
        else:
            # Try to infer winner from policy count
            liberal_policies = 0
            fascist_policies = 0
            
            for entry in entries:
                if entry.get('event_type') == 'POLICY_ENACTED':
                    policy = entry.get('data', {}).get('policy', '').lower()
                    if 'liberal' in policy:
                        liberal_policies += 1
                    elif 'fascist' in policy:
                        fascist_policies += 1
            
            if liberal_policies >= 5:
                winner = 'liberals'
                win_reason = '5 liberal policies'
            elif fascist_policies >= 6:
                winner = 'fascists'
                win_reason = '6 fascist policies'
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
            pid = int(player_key.split('_p')[-1])
            
            # Check if Hitler survived (only relevant for Hitler role)
            hitler_survived = False
            if role == 'hitler':
                # Check if Hitler was executed
                was_executed = False
                for entry in entries:
                    if entry.get('event_type') == 'EXECUTION':
                        executed_id = entry.get('data', {}).get('executed')
                        if executed_id == pid:
                            was_executed = True
                            break
                hitler_survived = not was_executed
            
            player_stats[player_key] = {
                'role': role,
                'hitler_survived': hitler_survived if role == 'hitler' else None,
            }
        
        return GameResult(
            game_id=game_id,
            winner=winner,
            win_reason=win_reason,
            duration=duration,
            num_rounds=rounds,
            players=list(player_roles.keys()),
            player_roles=player_roles,
            player_stats=player_stats,
        )
    
    def _is_winner(self, player: str, result: GameResult) -> bool:
        """Check if player won (team-based)."""
        role = result.player_roles.get(player)
        if not role:
            return False
        
        # Match winner with team
        winner = result.winner.lower() if result.winner else ''
        
        # Liberals win together
        if 'liberal' in winner:
            return role == 'liberal'
        # Fascists and Hitler win together
        elif 'fascist' in winner:
            return role in ['fascist', 'hitler']
        
        return False
    
    def generate_summary_table(self, include_incomplete: bool = False) -> pd.DataFrame:
        """Generate Secret Hitler-specific summary table with detailed columns.
        
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
            'liberal_games': 0,
            'liberal_wins': 0,
            'fascist_games': 0,  # Includes both fascist and hitler roles
            'fascist_wins': 0,
            'hitler_games': 0,
            'hitler_survived': 0,
        })
        
        for result in results_to_analyze:
            for player_key, role in result.player_roles.items():
                if not role:
                    continue
                
                # Extract model name (remove _pX suffix)
                model = '_'.join(player_key.split('_')[:-1]) if '_p' in player_key else player_key
                stats = model_stats[model]
                
                # Count games
                stats['games'] += 1
                
                # Track role-based games
                if role == 'liberal':
                    stats['liberal_games'] += 1
                elif role in ['fascist', 'hitler']:
                    stats['fascist_games'] += 1
                    
                    if role == 'hitler':
                        stats['hitler_games'] += 1
                        # Track Hitler survival
                        if result.player_stats.get(player_key, {}).get('hitler_survived', False):
                            stats['hitler_survived'] += 1
                
                # Check if winner
                is_winner = self._is_winner(player_key, result)
                if is_winner:
                    stats['wins'] += 1
                    
                    if role == 'liberal':
                        stats['liberal_wins'] += 1
                    elif role in ['fascist', 'hitler']:
                        stats['fascist_wins'] += 1
        
        # Build dataframe
        rows = []
        for model in sorted(model_stats.keys()):
            stats = model_stats[model]
            
            # Calculate metrics
            overall_wr = (stats['wins'] / stats['games'] * 100) if stats['games'] > 0 else 0
            lib_wr = (stats['liberal_wins'] / stats['liberal_games'] * 100) if stats['liberal_games'] > 0 else 0
            fasc_wr = (stats['fascist_wins'] / stats['fascist_games'] * 100) if stats['fascist_games'] > 0 else 0
            hitler_surv = (stats['hitler_survived'] / stats['hitler_games'] * 100) if stats['hitler_games'] > 0 else 0
            
            rows.append({
                'Model': model,
                'Games': stats['games'],
                'Overall WR': f"{overall_wr:.1f}%",
                'As Liberal': stats['liberal_games'],
                'Lib Win %': f"{lib_wr:.1f}%",
                'As Fascist': stats['fascist_games'],
                'Fasc Win %': f"{fasc_wr:.1f}%",
                'Hitler Surv%': f"{hitler_surv:.1f}%",
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
        """Generate Secret Hitler-specific statistics."""
        role_stats = defaultdict(lambda: defaultdict(int))
        
        for result in self.results:
            for player, role in result.player_roles.items():
                if not role:
                    continue
                    
                role_stats[player][f'{role}_games'] += 1
                
                if self._is_winner(player, result):
                    role_stats[player][f'{role}_wins'] += 1
        
        # Calculate role-specific metrics
        detailed = {}
        for player, stats in role_stats.items():
            detailed[player] = dict(stats)
            
            # Calculate win rates by role
            for role in ['liberal', 'fascist', 'hitler']:
                games = stats.get(f'{role}_games', 0)
                if games > 0:
                    wins = stats.get(f'{role}_wins', 0)
                    detailed[player][f'{role}_win_rate'] = wins / games
        
        return detailed


def main():
    """Run Secret Hitler evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Secret Hitler tournament')
    parser.add_argument('tournament_dir', type=Path, help='Tournament directory')
    parser.add_argument('--output', type=Path, help='Output directory (default: tournament_dir)')
    args = parser.parse_args()
    
    evaluator = SecretHitlerEvaluator(args.tournament_dir)
    evaluator.load_all_games()
    evaluator.print_summary()
    evaluator.save_tables(args.output)


if __name__ == '__main__':
    main()

