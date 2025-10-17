"""Example: Run Werewolf game with LLM agents."""

import asyncio
import os
import yaml
from pathlib import Path

from sdb.environments.werewolf import WerewolfEnv, WerewolfConfig
from sdb.agents.llm.openrouter_agent import OpenRouterAgent
from sdb.logging.game_logger import GameLogger


async def main():
    """Run Werewolf game with LLM agents."""
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    print("🐺 Werewolf")
    print("=" * 70)
    
    # Load configuration from YAML
    config_path = Path(__file__).parent.parent / "configs" / "werewolf.yaml"
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    print(f"✅ Loaded config from werewolf.yaml")
    
    # Extract game config (filter to only WerewolfConfig parameters)
    game_config = {
        'n_players': yaml_config['n_players'],
        'n_werewolves': yaml_config['n_werewolves'],
        'include_seer': yaml_config['include_seer'],
        'include_doctor': yaml_config['include_doctor'],
        'max_debate_turns': yaml_config.get('debate_turns', 5),  # YAML uses 'debate_turns', config uses 'max_debate_turns'
        'max_rounds': yaml_config.get('max_rounds', 50),
        'vote_requires_majority': yaml_config.get('vote_requires_majority', True),
    }
    
    config = WerewolfConfig(**game_config)
    
    print(f"\n⚙️  Configuration:")
    print(f"  • Players: {config.n_players}")
    print(f"  • Werewolves: {config.n_werewolves}")
    special_roles = []
    if config.include_seer:
        special_roles.append("Seer")
    if config.include_doctor:
        special_roles.append("Doctor")
    print(f"  • Special Roles: {', '.join(special_roles) if special_roles else 'None'}")
    print(f"  • Max debate turns per day: {config.max_debate_turns}")
    print(f"  • Max rounds: {config.max_rounds}")
    
    # Create agents
    agent_config = yaml_config.get('agent', {})
    model = agent_config.get('model', 'openai/gpt-4o-mini')
    temperature = agent_config.get('temperature', 0.8)
    memory_capacity = agent_config.get('memory_capacity', 40)
    
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
    output_dir = Path(yaml_config.get('logging', {}).get('output_dir', 'experiments/werewolf'))
    logger = GameLogger(output_dir=output_dir, log_private=True)
    print(f"\n📁 Logs will be saved to: {output_dir}/ (including private info)")
    
    # Create environment
    env = WerewolfEnv(agents=agents, config=config, logger=logger)
    
    print(f"\n🎲 Game Started!")
    print(f"\n👥 Roles:")
    for player in env.state.players.values():
        role_emoji = "🐺" if player.role.value == "werewolf" else "👤"
        print(f"  {role_emoji} Player {player.player_id}: {player.role.value}")
    
    print(f"\n▶️  Game will run automatically...")
    print("   This may take several minutes due to multiple phases per round.")
    print("-" * 70)
    
    # Run game with max rounds limit
    round_count = 0
    max_rounds = 100
    
    while not env.state.winner and round_count < max_rounds:
        round_count += 1
        
        if round_count % 10 == 0:
            print(f"  ... Round {round_count}, {len(env.state.get_alive_players())} players alive")
        
        # Get observations
        obs = env._get_observations()
        
        # Collect all players who need to act
        act_players = [
            (player_id, observation)
            for player_id, observation in obs.items()
            if observation.data.get("type") == "act" and observation.data.get("instruction")
        ]
        
        # Call all agents in parallel using asyncio.gather
        actions = {}
        if act_players:
            tasks = [agents[pid].act_async(observation) for pid, observation in act_players]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful actions (filter out exceptions)
            for (pid, _), result in zip(act_players, results):
                if not isinstance(result, Exception):
                    actions[pid] = result
                # Silently ignore exceptions for cleaner output
        
        # Step environment
        if actions:
            obs, rewards, done, info = env.step(actions)
        
        if env.state.winner:
            break
    
    # Show results
    print("\n" + "=" * 70)
    print("🏁 GAME OVER!")
    print("=" * 70)
    
    winner = env.get_winner()
    win_reason = env.get_win_reason()
    
    if winner:
        winner_emoji = "🐺" if winner == "werewolves" else "👥"
        print(f"\n🏆 Winner: {winner_emoji} {winner.upper()}")
    else:
        print(f"\n🏆 Winner: ⚠️ NO WINNER (game may have ended unexpectedly)")
    
    if win_reason:
        print(f"📝 Win Condition: {win_reason}")
    else:
        print(f"📝 Win Condition: Game reached max rounds without conclusion")
    
    alive_players = env.state.get_alive_players()
    print(f"\n📊 Final Stats:")
    print(f"  • Total rounds: {round_count}")
    print(f"  • Players remaining: {len(alive_players)}")
    print(f"  • Night phases: {len(env.state.night_results)}")
    print(f"  • Day phases: {len(env.state.day_results)}")
    
    print(f"\n📁 Game log: {output_dir}/{logger.game_id}.jsonl")
    print(f"  ({len(logger.entries)} events logged)")
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())

