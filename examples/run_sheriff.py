"""Example: Run Sheriff of Nottingham game with LLM agents."""

import asyncio
import os
import yaml
from pathlib import Path

from sdb.environments.sheriff import SheriffEnv, SheriffConfig
from sdb.agents.llm.openrouter_agent import OpenRouterAgent
from sdb.logging.game_logger import GameLogger


async def main():
    """Run Sheriff of Nottingham game with LLM agents."""
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    print("🏹 Sheriff of Nottingham")
    print("=" * 70)
    
    # Load configuration from YAML
    config_path = Path("configs/sheriff.yaml")
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Game configuration - map YAML keys to SheriffConfig parameters
    game_config = {
        'n_players': yaml_config.get('n_players', 4),
        'sheriff_rotations': yaml_config.get('n_rounds', 2),  # YAML uses 'n_rounds'
        'bag_limit': yaml_config.get('max_bag_size', 5),  # YAML uses 'max_bag_size'
        'max_negotiation_rounds': yaml_config.get('negotiation_rounds', 2),
        'include_royal': yaml_config.get('include_royal', False),
        'hand_size': yaml_config.get('hand_size', 6),
    }
    config = SheriffConfig(**game_config)
    
    # Store extra config values not in SheriffConfig
    starting_gold = yaml_config.get('starting_gold', 50)
    
    # Agent configuration
    agent_config = yaml_config.get('agent', {})
    model = agent_config.get('model', 'anthropic/claude-3.5-sonnet')
    temperature = agent_config.get('temperature', 0.85)
    memory_capacity = agent_config.get('memory_capacity', 30)
    
    # Logging configuration
    logging_config = yaml_config.get('logging', {})
    output_dir = Path(logging_config.get('output_dir', 'experiments/sheriff'))
    
    print(f"\n⚙️  Configuration:")
    print(f"  • Players: {config.n_players}")
    print(f"  • Sheriff rotations: {config.sheriff_rotations}")
    print(f"  • Starting gold: {starting_gold}")
    print(f"  • Max bag size: {config.bag_limit}")
    print(f"  • Max negotiation rounds: {config.max_negotiation_rounds}")
    print(f"  • Features: Bluffing, negotiation, bribery, inspection")
    
    # Create agents
    print(f"\n🤖 Creating {config.n_players} LLM agents ({model})...")
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
    
    # Setup logging
    logger = GameLogger(output_dir=output_dir)
    print(f"\n📁 Logs will be saved to: {output_dir}/")
    
    # Create environment (automatically calls reset())
    env = SheriffEnv(agents=agents, config=config, logger=logger)
    
    # Show initial state
    print(f"\n🎲 Game Started!")
    print(f"💰 Each player starts with {starting_gold} gold")
    
    print(f"\n▶️  Game running (this may take several minutes)...")
    print("-" * 70)
    
    # Run game
    round_count = 0
    max_rounds = 200
    
    while not env.state.game_over and round_count < max_rounds:
        round_count += 1
        
        # Show sheriff rotations
        if env.state.phase.value == "market" and env.state.round_step == env.state.get_merchant_idx(0):
            rotation = sum(env.state.rotation_counts)
            print(f"\n🏹 Sheriff Rotation {rotation+1}: Player {env.state.sheriff_idx} is Sheriff")
        
        # Get observations
        obs = env._get_observations()
        
        # Determine which player(s) should act based on current phase
        from sdb.environments.sheriff.types import Phase
        from sdb.core.types import Action
        
        actions = {}
        active_players = []
        
        if env.state.phase in [Phase.MARKET, Phase.LOAD, Phase.DECLARE]:
            # Only the current merchant acts
            active_players = [env.state.round_step]
        elif env.state.phase == Phase.NEGOTIATE:
            # Current round_step indicates who should act (merchants or sheriff)
            active_players = [env.state.round_step]
        elif env.state.phase == Phase.INSPECT:
            # Only sheriff acts
            active_players = [env.state.sheriff_idx]
        elif env.state.phase == Phase.RESOLVE:
            # System phase - call step with empty actions to trigger resolve
            active_players = []
            actions = {}  # Empty actions dict
        
        # Debug print
        if round_count == 1:
            print(f"  🔍 Initial phase: {env.state.phase.value}, active player: {active_players}")
        
        # Collect actions from active players
        for player_id in active_players:
            if player_id in obs:
                try:
                    action = await agents[player_id].act_async(obs[player_id])
                    actions[player_id] = action
                    
                    # Show phase transitions
                    action_type = action.data.get("type", "")
                    if env.state.phase == Phase.MARKET:
                        print(f"  🛒 Player {player_id} draws from market (hand: {len(env.state.get_player(player_id).hand)} cards)")
                    elif action_type == "load":
                        bag_size = len(action.data.get("card_ids", []))
                        print(f"  📦 Player {player_id} loads {bag_size} cards into bag")
                    elif action_type == "declare":
                        dtype = action.data.get("declared_type", "?")
                        dcount = action.data.get("declared_count", 0)
                        print(f"  🗣️  Player {player_id} declares: {dcount} {dtype}")
                    elif action_type == "offer":
                        gold = action.data.get("gold", 0)
                        if gold > 0:
                            print(f"  💰 Player {player_id} offers {gold} gold bribe")
                    elif action_type == "inspect":
                        target = action.data.get("merchant", action.data.get("merchant_id", -1))
                        choice = action.data.get("choice", "pass")
                        if choice == "inspect":
                            print(f"  🔍 Sheriff inspects Player {target}'s bag")
                        else:
                            print(f"  👋 Sheriff lets Player {target} pass")
                        
                except Exception as e:
                    print(f"  ❌ Error from Player {player_id}: {str(e)}")
                    # Provide fallback action based on phase
                    if env.state.phase == Phase.MARKET:
                        actions[player_id] = Action(player_id=player_id, data={"type": "draw", "source": "deck", "count": 1, "discard_ids": []})
                    elif env.state.phase == Phase.LOAD:
                        # Load some cards from hand
                        hand = env.state.get_player(player_id).hand
                        cards_to_load = hand[:min(3, len(hand))]
                        actions[player_id] = Action(player_id=player_id, data={"type": "load", "card_ids": cards_to_load})
                    elif env.state.phase == Phase.DECLARE:
                        bag_size = len(env.state.get_player(player_id).bag)
                        actions[player_id] = Action(player_id=player_id, data={"type": "declare", "declared_type": "apples", "declared_count": bag_size})
                    else:
                        actions[player_id] = Action(player_id=player_id, data={"type": "wait"})
        
        # Step environment
        # For RESOLVE phase: call step even with empty actions (system phase)
        # For other phases: only call if we have actions from active players
        should_step = (env.state.phase == Phase.RESOLVE) or (active_players and actions)
        
        if should_step:
            try:
                obs, rewards, done, info = env.step(actions)
                
                # Show rotation after resolve
                if env.state.phase == Phase.MARKET and round_count > 1:
                    print(f"\n🔄 Sheriff rotated! New sheriff: Player {env.state.sheriff_idx}")
            except Exception as e:
                print(f"  ⚠️  Step error: {str(e)}")
                break
        
        if env.state.game_over:
            break
    
    # Show results
    print("\n" + "=" * 70)
    print("🏁 GAME OVER!")
    print("=" * 70)
    
    winner = env.get_winner()
    win_reason = env.get_win_reason()
    
    if winner is not None:
        print(f"\n🏆 Winner: Player {winner}")
    else:
        print(f"\n🏆 Winner: DRAW")
    
    if win_reason:
        print(f"📝 Win Condition: {win_reason}")
    else:
        print(f"📝 Game ended")
    
    print(f"\n💰 Final Gold:")
    for player in env.state.players:
        print(f"  Player {player.pid}: {player.gold} gold")
    
    print(f"\n📊 Game Stats:")
    print(f"  • Total sheriff rotations: {sum(env.state.rotation_counts)}")
    print(f"  • Total actions: {round_count}")
    
    print(f"\n📁 Game log: {output_dir}/{logger.game_id}.jsonl")
    print(f"  ({len(logger.entries)} events logged)")
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())
