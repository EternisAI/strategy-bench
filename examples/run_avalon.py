"""Example: Run Avalon game with LLM agents."""

import asyncio
import os
from pathlib import Path

from sdb.environments.avalon import AvalonEnv, AvalonConfig
from sdb.agents.llm.openrouter_agent import OpenRouterAgent
from sdb.logging.game_logger import GameLogger


async def main():
    """Run Avalon game with LLM agents."""
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    print("🎮 Avalon - The Resistance")
    print("=" * 70)
    
    # Load configuration from YAML
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "configs" / "avalon.yaml"
    
    if not config_path.exists():
        print(f"⚠️  Config file not found, using defaults")
        config = AvalonConfig(
            n_players=5,
            include_merlin=True,
            include_percival=False,
            include_morgana=True,
            include_mordred=False,
            include_oberon=False,
        )
    else:
        import yaml
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
        
        # Extract only valid AvalonConfig parameters
        valid_params = ['n_players', 'seed', 'roles', 'include_merlin', 
                       'include_percival', 'include_morgana', 'include_mordred', 'include_oberon']
        game_config = {k: v for k, v in yaml_config.items() if k in valid_params}
        
        config = AvalonConfig(**game_config)
        print(f"✅ Loaded config from {config_path.name}")
    
    # Use config for agent settings if available
    if config_path.exists():
        agent_config = yaml_config.get('agent', {})
        model = agent_config.get('model', 'anthropic/claude-3.5-sonnet')
        temperature = agent_config.get('temperature', 0.8)
        memory_capacity = agent_config.get('memory_capacity', 30)
    else:
        model = 'anthropic/claude-3.5-sonnet'
        temperature = 0.8
        memory_capacity = 30
    
    
    print(f"\n⚙️  Configuration:")
    print(f"  • Players: {config.n_players}")
    
    # Show actual special roles enabled
    special_roles = []
    if config.include_merlin:
        special_roles.append("Merlin")
    if config.include_percival:
        special_roles.append("Percival")
    if config.include_morgana:
        special_roles.append("Morgana")
    if config.include_mordred:
        special_roles.append("Mordred")
    if config.include_oberon:
        special_roles.append("Oberon")
    
    # Assassin is always present in Avalon
    special_roles.append("Assassin")
    
    print(f"  • Special Roles: {', '.join(special_roles) if special_roles else 'None (basic roles only)'}")
    print(f"  • Features: Team discussion, team voting, quest voting, assassination")
    
    # Create agents
    print(f"\n🤖 Creating {config.n_players} LLM agents...")
    agents = []
    for i in range(config.n_players):
        agent = OpenRouterAgent(
            player_id=i,
            model=model,
            temperature=temperature,
            memory_capacity=memory_capacity
        )
        agents.append(agent)
    print(f"  ✅ All agents created (model: {model})")
    
    # Setup logging
    if config_path.exists():
        logging_config = yaml_config.get('logging', {})
        output_dir = Path(logging_config.get('output_dir', 'experiments/avalon_example'))
    else:
        output_dir = Path("experiments/avalon_example")
    logger = GameLogger(output_dir=output_dir)
    print(f"\n📁 Logs will be saved to: {output_dir}/")
    
    # Create environment (reset is called automatically in __init__)
    env = AvalonEnv(agents=agents, config=config, logger=logger)
    
    # Show initial state
    print(f"\n🎲 Game Started!")
    print(f"\n👥 Roles assigned:")
    for i, player in enumerate(env.state.players):
        team_emoji = "⚔️" if player.team.value == "good" else "🗡️"
        print(f"  {team_emoji} Player {i}: {player.role.value}")
    
    # Run game
    print(f"\n▶️  Starting gameplay...")
    print("-" * 70)
    
    round_count = 0
    max_rounds = 200  # Safety limit for game loop iterations
    
    while not env.state.game_over and round_count < max_rounds:
        round_count += 1
        
        # Get current phase
        phase = env.state.current_phase.value
        
        # Show phase transitions
        if round_count == 1 or phase != getattr(main, 'last_phase', None):
            print(f"\n📍 Phase: {phase.upper()}")
            if phase == "team_discussion":
                print(f"   💬 Players discussing Quest {env.state.current_quest + 1} team proposal")
            elif phase == "team_voting":
                print(f"   🗳️  Voting on team for Quest {env.state.current_quest + 1}")
            elif phase == "quest_voting":
                team_size = len(env.state.current_proposal.team) if env.state.current_proposal else 0
                print(f"   ⚔️  Quest {env.state.current_quest + 1} in progress (collecting {team_size} anonymous ballots...)")
        main.last_phase = phase
        
        # Get observations and have agents act
        obs = env._get_observations()
        actions = {}
        
        # Filter which players should act based on phase
        players_to_act = list(obs.keys())
        
        if phase == "team_selection":
            # Only quest leader proposes
            players_to_act = [env.state.quest_leader]
        elif phase == "quest_voting" and env.state.current_proposal:
            # Only team members vote on quests
            players_to_act = [pid for pid in players_to_act if pid in env.state.current_proposal.team]
        # For team_voting and team_discussion, all players act (discussion is sequential, handled by env)
        
        for player_id in players_to_act:
            if player_id in obs and obs[player_id].data.get("instruction"):
                try:
                    action = await agents[player_id].act_async(obs[player_id])
                    actions[player_id] = action
                    
                    # Only show actions that match the current phase
                    action_type = action.data.get("type", "")
                    
                    # Team discussion: only show discuss actions
                    if phase == "team_discussion" and action_type in ["discuss_team", "discuss"]:
                        stmt = action.data.get("statement", "")
                        if stmt:
                            print(f"  💭 Player {player_id}: \"{stmt[:60]}...\"")
                    
                    # Team selection: only show propose from quest leader
                    elif phase == "team_selection" and action_type in ["propose_team", "propose"]:
                        if player_id == env.state.quest_leader:
                            team = action.data.get("team", [])
                            # Show proposal ID if available
                            proposal_id = f"#{env.state.total_proposals + 1}" if hasattr(env.state, 'total_proposals') else ""
                            quest_label = f"Q{env.state.current_quest + 1}"
                            print(f"  👥 {proposal_id} ({quest_label}) Player {player_id} proposes team: {team}")
                    
                    # Team voting: show votes (quest votes are anonymous, shown in summary only)
                    elif phase == "team_voting" and action_type in ["vote", "team_vote"]:
                        vote = action.data.get("vote")
                        if vote:
                            emoji = "✅" if vote == "approve" else "❌"
                            print(f"  {emoji} Player {player_id} votes {vote}")
                    
                    # Quest voting: don't show individual ballots (anonymous!)
                    # Progress is shown via quest_voters_done tracking in env
                        
                except Exception as e:
                    print(f"  ⚠️  Player {player_id} error: {e}")
        
        # Step environment
        old_quest_count = len(env.state.quest_results)
        old_proposal_count = len(env.state.proposal_history)
        
        if actions:
            obs, rewards, done, info = env.step(actions)
        
        # Check for new quest result
        if len(env.state.quest_results) > old_quest_count:
            result = env.state.quest_results[-1]
            emoji = "✅" if result.succeeded else "❌"
            outcome = "SUCCESS" if result.succeeded else "FAIL"
            print(f"\n📣 Quest {result.quest_num + 1}: {emoji} {outcome} ({result.success_votes}S/{result.fail_votes}F)")
        
        # Check for new team vote result
        if len(env.state.proposal_history) > old_proposal_count:
            proposal = env.state.proposal_history[-1]
            if hasattr(proposal, 'votes') and proposal.votes:
                approves = sum(1 for v in proposal.votes.values() if v == "approve")
                rejects = len(proposal.votes) - approves
                emoji = "✅" if proposal.approved else "❌"
                outcome = "APPROVED" if proposal.approved else "REJECTED"
                quest_label = f"Q{proposal.quest_num + 1}" if hasattr(proposal, 'quest_num') else ""
                prop_id = f"#{proposal.proposal_idx}" if hasattr(proposal, 'proposal_idx') else ""
                print(f"🧾 Proposal {prop_id} ({quest_label}) — Approve {approves} / Reject {rejects} → {emoji} {outcome}")
                print(f"   Team: {proposal.team}")
        
        if env.state.game_over:
            break
    
    # Check if game hit max rounds without completing
    if round_count >= max_rounds and not env.state.game_over:
        print(f"\n⚠️  Game reached max rounds ({max_rounds}) without completing!")
        env.state.game_over = True
    
    # Show results
    print("\n" + "=" * 70)
    print("🏁 GAME OVER!")
    print("=" * 70)
    
    winner = env.get_winner()
    win_reason = env.get_win_reason()
    
    if winner:
        winner_emoji = "⚔️" if winner == "good" else "🗡️"
        print(f"\n🏆 Winner: {winner_emoji} {winner.upper()}")
    else:
        print(f"\n🏆 Winner: ⚠️ NO WINNER (game may have ended unexpectedly)")
    
    if win_reason:
        print(f"📝 Win Condition: {win_reason}")
    else:
        print(f"📝 Win Condition: Unknown")
    
    print(f"\n📊 Final Stats:")
    print(f"  • Total rounds: {round_count}")
    print(f"  • Quests succeeded: {env.state.quests_succeeded}")
    print(f"  • Quests failed: {env.state.quests_failed}")
    print(f"  • Team rejections: {env.state.team_rejections}")
    
    print(f"\n📁 Game log: {output_dir}/{logger.game_id}.jsonl")
    print(f"  ({len(logger.entries)} events logged)")
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())

