"""Evaluator for Spyfall tournaments."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from evaluations.base_evaluator import BaseEvaluator, GameResult, read_jsonl, extract_model_name


class SpyfallEvaluator(BaseEvaluator):
    """Evaluator for Spyfall game tournaments."""
    
    def parse_game_log(self, log_file: Path) -> Optional[GameResult]:
        """Parse Spyfall game log.
        
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
        player_id_mapping = {}  # pid -> (model, role)
        
        # Find agent metadata - can be in data.event or data.action
        agents_info = next((e for e in entries 
                           if e.get('data', {}).get('event') == 'agent_metadata' 
                           or e.get('data', {}).get('action') == 'agent_metadata'), None)
        
        # Find role assignment - contains spy_index
        role_assignment = next((e for e in entries 
                               if e.get('data', {}).get('action') == 'role_assignment'), None)
        
        if not agents_info or not role_assignment:
            return None
        
        agents = agents_info['data']['agents']
        spy_index = role_assignment['data'].get('spy_index')
        
        # Map all players to roles
        for pid, agent_data in agents.items():
            model = extract_model_name(agent_data['model'])
            pid_int = int(pid)
            role = 'spy' if pid_int == spy_index else 'non-spy'
            
            player_key = f"{model}_p{pid}"
            player_roles[player_key] = role
            player_id_mapping[pid_int] = (model, role)
        
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
        
        # Player stats
        player_stats = {}
        for pid_int, (model, role) in player_id_mapping.items():
            player_key = f"{model}_p{pid_int}"
            player_stats[player_key] = {
                'role': role,
                'model': model,
                'questions_asked': self._count_questions_asked(pid_int, entries),
                'questions_answered': self._count_questions_answered(pid_int, entries),
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
    
    def _count_questions_asked(self, player_id: int, entries: List[Dict]) -> int:
        """Count questions asked by player."""
        questions = [e for e in entries 
                    if e.get('event_type') == 'PLAYER_ACTION'
                    and e.get('player_id') == player_id
                    and e.get('data', {}).get('action_type') == 'ask_question']
        
        return len(questions)
    
    def _count_questions_answered(self, player_id: int, entries: List[Dict]) -> int:
        """Count questions answered by player."""
        answers = [e for e in entries 
                  if e.get('event_type') == 'PLAYER_ACTION'
                  and e.get('data', {}).get('target') == player_id
                  and e.get('data', {}).get('action_type') == 'answer_question']
        
        return len(answers)
    
    def _is_winner(self, player: str, result: GameResult) -> bool:
        """Check if player won (team-based)."""
        role = result.player_roles.get(player)
        if not role:
            return False
        
        # Spy wins or non-spies win
        if result.winner == 'spy':
            return role == 'spy'
        elif result.winner == 'non-spies' or result.winner == 'non-spy':
            return role == 'non-spy'
        
        return False
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate Spyfall summary table.
        
        Columns:
        - Model: Model name
        - Games: Total games played
        - Overall WR: Overall win rate
        - Spy Games: Games as spy
        - Spy WR: Win rate as spy
        - Non-Spy Games: Games as non-spy
        - Non-Spy WR: Win rate as non-spy
        - Avg Q Asked: Average questions asked
        - Avg Q Ans: Average questions answered
        """
        if not self.results:
            raise ValueError("No results loaded. Call load_all_games() first.")
        
        model_stats = defaultdict(lambda: {
            'games': 0,
            'wins': 0,
            'spy_games': 0,
            'spy_wins': 0,
            'nonspy_games': 0,
            'nonspy_wins': 0,
            'questions_asked': 0,
            'questions_answered': 0,
        })
        
        for result in self.results:
            for player_key, role in result.player_roles.items():
                if not role:
                    continue
                
                # Extract model from player_key (format: "model_pX")
                model = '_'.join(player_key.split('_')[:-1]) if '_p' in player_key else player_key
                stats = model_stats[model]
                
                stats['games'] += 1
                
                # Track role-specific stats
                if role == 'spy':
                    stats['spy_games'] += 1
                elif role == 'non-spy':
                    stats['nonspy_games'] += 1
                
                # Check if won
                is_winner = self._is_winner(player_key, result)
                if is_winner:
                    stats['wins'] += 1
                    if role == 'spy':
                        stats['spy_wins'] += 1
                    elif role == 'non-spy':
                        stats['nonspy_wins'] += 1
                
                # Count questions
                player_stats = result.player_stats.get(player_key, {})
                stats['questions_asked'] += player_stats.get('questions_asked', 0)
                stats['questions_answered'] += player_stats.get('questions_answered', 0)
        
        rows = []
        for model in sorted(model_stats.keys()):
            stats = model_stats[model]
            
            overall_wr = (stats['wins'] / stats['games'] * 100) if stats['games'] > 0 else 0
            spy_wr = (stats['spy_wins'] / stats['spy_games'] * 100) if stats['spy_games'] > 0 else 0
            nonspy_wr = (stats['nonspy_wins'] / stats['nonspy_games'] * 100) if stats['nonspy_games'] > 0 else 0
            
            avg_q_asked = stats['questions_asked'] / stats['games'] if stats['games'] > 0 else 0
            avg_q_ans = stats['questions_answered'] / stats['games'] if stats['games'] > 0 else 0
            
            rows.append({
                'Model': model,
                'Games': stats['games'],
                'Overall WR': f"{overall_wr:.1f}%",
                'Spy Games': stats['spy_games'],
                'Spy WR': f"{spy_wr:.1f}%",
                'Non-Spy Games': stats['nonspy_games'],
                'Non-Spy WR': f"{nonspy_wr:.1f}%",
                'Avg Q Asked': f"{avg_q_asked:.1f}",
                'Avg Q Ans': f"{avg_q_ans:.1f}",
            })
        
        return pd.DataFrame(rows)
    
    def generate_detailed_stats(self) -> Dict[str, Any]:
        """Generate Spyfall-specific statistics."""
        role_stats = defaultdict(lambda: defaultdict(int))
        
        for result in self.results:
            for player, role in result.player_roles.items():
                if not role:
                    continue
                    
                role_stats[player][f'{role}_games'] += 1
                
                if self._is_winner(player, result):
                    role_stats[player][f'{role}_wins'] += 1
                
                # Question participation
                stats = result.player_stats.get(player, {})
                role_stats[player][f'{role}_questions_asked'] += stats.get('questions_asked', 0)
                role_stats[player][f'{role}_questions_answered'] += stats.get('questions_answered', 0)
        
        # Calculate role-specific metrics
        detailed = {}
        for player, stats in role_stats.items():
            detailed[player] = dict(stats)
            
            # Calculate win rates by role
            for role in ['spy', 'non-spy']:
                games = stats.get(f'{role}_games', 0)
                if games > 0:
                    wins = stats.get(f'{role}_wins', 0)
                    detailed[player][f'{role}_win_rate'] = wins / games
                    
                    # Average questions
                    questions_asked = stats.get(f'{role}_questions_asked', 0)
                    questions_answered = stats.get(f'{role}_questions_answered', 0)
                    detailed[player][f'{role}_avg_questions_asked'] = questions_asked / games
                    detailed[player][f'{role}_avg_questions_answered'] = questions_answered / games
        
        return detailed


def main():
    """Run Spyfall evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Spyfall tournament')
    parser.add_argument('tournament_dir', type=Path, help='Tournament directory')
    parser.add_argument('--output', type=Path, help='Output directory (default: tournament_dir)')
    args = parser.parse_args()
    
    evaluator = SpyfallEvaluator(args.tournament_dir)
    evaluator.load_all_games()
    evaluator.print_summary()
    evaluator.save_tables(args.output)


if __name__ == '__main__':
    main()

