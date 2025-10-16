"""Example: Run Among Us game with LLM agents."""

import asyncio
import os
from pathlib import Path
import yaml

from sdb.environments.among_us import AmongUsEnv, AmongUsConfig
from sdb.agents.llm.openrouter_agent import OpenRouterAgent
from sdb.logging.game_logger import GameLogger


async def main():
    """Run Among Us game with LLM agents."""
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    print("🚀 Among Us")
    print("=" * 70)
    
    # Load configuration from YAML
    config_path = Path(__file__).parent.parent / "configs" / "among_us.yaml"
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    print(f"✅ Loaded config from among_us.yaml")
    
    # Extract game config (filter to only AmongUsConfig parameters)
    game_config = {
        'n_players': yaml_config['n_players'],
        'n_impostors': yaml_config['n_impostors'],
        'tasks_per_player': yaml_config.get('n_tasks_per_player', 5),  # YAML uses n_tasks_per_player
        'discussion_rounds': yaml_config.get('discussion_rounds', 2),
        'max_task_rounds': yaml_config.get('max_task_rounds', 50),
        'kill_cooldown': yaml_config.get('kill_cooldown', 3),
        'emergency_meetings': yaml_config.get('emergency_meetings', 1),
    }
    
    config = AmongUsConfig(**game_config)
    
    print(f"\n⚙️  Configuration:")
    print(f"  • Players: {config.n_players}")
    print(f"  • Impostors: {config.n_impostors}")
    print(f"  • Tasks per player: {config.tasks_per_player}")
    print(f"  • Discussion rounds: {config.discussion_rounds}")
    print(f"  • Max task rounds: {config.max_task_rounds}")
    print(f"  • Kill cooldown: {config.kill_cooldown}")
    print(f"  • Features: Spatial map, movement, tasks, emergency meetings")
    
    # Create agents from YAML config
    agent_config = yaml_config.get('agent', {})
    model = agent_config.get('model', 'anthropic/claude-3.5-sonnet')
    temperature = agent_config.get('temperature', 0.8)
    memory_capacity = agent_config.get('memory_capacity', 35)
    
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
    print("  ✅ All agents created")
    
    # Setup logging from YAML config
    logging_config = yaml_config.get('logging', {})
    output_dir = Path(logging_config.get('output_dir', 'experiments/among_us'))
    logger = GameLogger(output_dir=output_dir)
    print(f"\n📁 Logs will be saved to: {output_dir}/")
    
    # Create environment (reset is called automatically in __init__)
    env = AmongUsEnv(agents=agents, config=config, logger=logger)
    
    # Show initial state
    print(f"\n🎲 Game Started!")
    print(f"\n🗺️  Spatial Map Active - Players start in Cafeteria")
    print(f"👥 Roles:")
    for i, player in env.state.players.items():
        role_emoji = "🔪" if player.role.value == "impostor" else "👷"
        print(f"  {role_emoji} Player {i}: {player.role.value}")
    
    print(f"\n▶️  Game running...")
    print("-" * 70)
    
    # Run game
    task_round = 0
    max_task_rounds = 100
    
    while not env.state.winner and task_round < max_task_rounds:
        task_round += 1
        
        # Show progress periodically
        if task_round % 10 == 0:
            alive = len([p for p in env.state.players.values() if p.is_alive])
            total_tasks = sum(p.tasks_completed for p in env.state.players.values())
            print(f"  Round {task_round}: {alive} alive, {total_tasks} tasks completed")
        
        # Get observations
        obs = env._get_observations()
        
        # Have agents act
        actions = {}
        for player_id, observation in obs.items():
            if observation.data.get("instruction"):
                try:
                    action = await agents[player_id].act_async(observation)
                    actions[player_id] = action
                    
                    # Show interesting spatial actions
                    action_type = action.data.get("type", "")
                    if action_type == "move" and task_round <= 5:
                        dest = action.data.get("room", action.data.get("destination"))
                        if dest:
                            print(f"  🚶 Player {player_id} moves to {dest}")
                    elif action_type == "vent" and task_round <= 5:
                        dest = action.data.get("room", action.data.get("destination"))
                        if dest:
                            print(f"  🕳️  Player {player_id} vents to {dest}")
                    elif action_type == "kill" and task_round <= 20:
                        target = action.data.get("target")
                        if target is not None and isinstance(target, int):
                            print(f"  🔪 Player {player_id} attempts to kill Player {target}")
                    elif action_type == "report" and task_round <= 20:
                        body = action.data.get("body_id")
                        if body is not None:
                            print(f"  🚨 Player {player_id} reports body {body}")
                        
                except Exception as e:
                    print(f"  ❌ Error from Player {player_id}: {str(e)}")
                    # Provide fallback wait action
                    from sdb.core.types import Action
                    actions[player_id] = Action(player_id=player_id, data={"type": "wait"})
        
        # Step environment - Among Us expects one action at a time
        if actions:
            try:
                for player_id, action in actions.items():
                    obs, rewards, done, info = env.step(action)
                    if done:
                        break
            except Exception as e:
                print(f"  ⚠️  Step error: {str(e)}")
                import traceback
                traceback.print_exc()
                break
        
        if env.state.winner:
            break
    
    # Show results
    print("\n" + "=" * 70)
    print("🏁 GAME OVER!")
    print("=" * 70)
    
    winner = env.get_winner()
    win_reason = env.get_win_reason()
    
    if winner:
        winner_emoji = "🔪" if winner == "impostors" else "👷"
        print(f"\n🏆 Winner: {winner_emoji} {winner.upper()}")
    else:
        print(f"\n🏆 Winner: DRAW (timeout)")
    
    if win_reason:
        print(f"📝 Win Condition: {win_reason}")
    else:
        print(f"📝 Game ended")
    
    total_tasks = sum(p.tasks_completed for p in env.state.players.values())
    alive_players = len([p for p in env.state.players.values() if p.is_alive])
    
    print(f"\n📊 Final Stats:")
    print(f"  • Task rounds: {task_round}")
    print(f"  • Players alive: {alive_players}/{config.n_players}")
    print(f"  • Tasks completed: {total_tasks}")
    print(f"  • Emergency meetings: {len(env.state.meeting_results)}")
    
    print(f"\n📁 Game log: {output_dir}/{logger.game_id}.jsonl")
    print(f"  ({len(logger.entries)} events logged)")
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())

