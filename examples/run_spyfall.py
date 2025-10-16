"""Example: Run Spyfall game with LLM agents."""

import asyncio
import os
import yaml
from pathlib import Path

from sdb.environments.spyfall import SpyfallEnv, SpyfallConfig
from sdb.agents.llm.openrouter_agent import OpenRouterAgent
from sdb.logging.game_logger import GameLogger


async def main():
    """Run Spyfall game with LLM agents."""
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    print("🕵️  Spyfall")
    print("=" * 70)
    
    # Load configuration from YAML
    config_path = Path(__file__).parent.parent / "configs" / "spyfall.yaml"
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    print(f"✅ Loaded config from spyfall.yaml")
    
    # Extract game config (filter to only SpyfallConfig parameters)
    game_config = {
        'n_players': yaml_config['n_players'],
        'max_turns': yaml_config['max_turns'],
    }
    
    config = SpyfallConfig(**game_config)
    
    # Note: timer_minutes is in yaml but not used in async implementation
    
    print(f"\n⚙️  Configuration:")
    print(f"  • Players: {config.n_players}")
    print(f"  • Max turns: {config.max_turns}")
    print(f"  • Features: Q&A rounds, location guessing, spy voting")
    
    # Create agents
    print(f"\n🤖 Creating {config.n_players} LLM agents...")
    agent_config = yaml_config.get('agent', {})
    agents = []
    for i in range(config.n_players):
        agent = OpenRouterAgent(
            player_id=i,
            model=agent_config.get('model', 'anthropic/claude-3.5-sonnet'),
            temperature=agent_config.get('temperature', 0.8),
            memory_capacity=agent_config.get('memory_capacity', 25)
        )
        agents.append(agent)
    print(f"  ✅ All agents created (model: {agent_config.get('model', 'anthropic/claude-3.5-sonnet')})")
    
    # Setup logging
    logging_config = yaml_config.get('logging', {})
    output_dir = Path(logging_config.get('output_dir', 'experiments/spyfall'))
    logger = GameLogger(output_dir=output_dir)
    print(f"\n📁 Logs will be saved to: {output_dir}/")
    
    # Create environment (automatically calls reset)
    env = SpyfallEnv(agents=agents, config=config, logger=logger)
    
    print(f"\n🎲 Game Started!")
    print(f"📍 Location: {env.state.location}")
    print(f"🕵️  The Spy is Player {env.state.spy_index} (hidden from others)")
    
    print(f"\n▶️  Starting Q&A rounds...")
    print("-" * 70)
    
    # Run game
    turn_count = 0
    max_rounds = 100  # Safety limit for game loop iterations
    
    while not env.state.winner and turn_count < max_rounds:
        turn_count += 1
        
        # Get observations
        obs = env._get_observations()
        
        # Debug: Check if any players can act
        act_players = [pid for pid, ob in obs.items() if ob.data.get("type") == "act"]
        if act_players and turn_count <= 5:  # Only print first few turns
            print(f"  📋 Players who can act: {act_players}")
        
        # Have agents act (only those with "act" observation type)
        actions = {}
        for player_id, observation in obs.items():
            obs_type = observation.data.get("type", "observe")
            instruction = observation.data.get("instruction", "")
            
            # Only ask agents to act if they have an "act" observation with instruction
            if obs_type == "act" and instruction:
                try:
                    action = await agents[player_id].act_async(observation)
                    actions[player_id] = action
                    
                    # Show Q&A interactions
                    action_type = action.data.get("type", "")
                    if action_type == "ask":
                        q = action.data.get("question", "")
                        target = action.data.get("target", -1)
                        print(f"  ❓ Player {player_id} asks Player {target}: \"{q[:70]}\"")
                    elif action_type == "answer":
                        ans = action.data.get("answer", "")
                        print(f"  💬 Player {player_id} answers: \"{ans[:70]}\"")
                    elif action_type == "spy_guess":
                        loc = action.data.get("guess", "")
                        print(f"  🎯 Player {player_id} (Spy) guesses: {loc}")
                    elif action_type == "accuse":
                        suspect = action.data.get("suspect", -1)
                        print(f"  🔍 Player {player_id} accuses Player {suspect} of being the spy!")
                    elif action_type == "nominate":
                        suspect = action.data.get("suspect", -1)
                        print(f"  📢 Player {player_id} nominates Player {suspect} as the spy")
                    elif action_type == "vote":
                        vote = action.data.get("vote", "")
                        print(f"  🗳️  Player {player_id} votes: {vote}")
                        
                except Exception as e:
                    print(f"  ⚠️  Player {player_id} error: {e}")
        
        # Step environment (always call, even with empty actions)
        if actions:
            obs, rewards, done, info = env.step(actions)
        else:
            # If no actions collected, something is wrong - log it
            print(f"  ⚠️  No actions collected on turn {turn_count}")
            # Still need to get observations for next iteration
            obs = env._get_observations()
        
        if env.state.winner:
            break
    
    # Check if game hit max rounds without completing
    if turn_count >= max_rounds and not env.state.winner:
        print(f"\n⚠️  Game reached max rounds ({max_rounds}) without completing!")
    
    # Show results
    print("\n" + "=" * 70)
    print("🏁 GAME OVER!")
    print("=" * 70)
    
    winner = env.get_winner()
    win_reason = env.get_win_reason()
    
    if winner:
        winner_emoji = "🕵️" if winner == "spy" else "👥"
        print(f"\n🏆 Winner: {winner_emoji} {winner.upper()}")
    else:
        print(f"\n🏆 Winner: ⚠️ NO WINNER (game may have ended unexpectedly)")
    
    if win_reason:
        print(f"📝 Win Condition: {win_reason}")
    else:
        print(f"📝 Win Condition: Game timed out")
    
    print(f"📍 The location was: {env.state.location}")
    print(f"🕵️  The spy was: Player {env.state.spy_index}")
    
    print(f"\n📊 Final Stats:")
    print(f"  • Q&A turns: {env.state.turn}")
    print(f"  • Questions asked: {len(env.state.qa_history)}")
    
    print(f"\n📁 Game log: {output_dir}/{logger.game_id}.jsonl")
    print(f"  ({len(logger.entries)} events logged)")
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())

