"""Example: Run Secret Hitler game with LLM agents."""

import os
from pathlib import Path
import yaml

from sdb.environments.secret_hitler import SecretHitlerEnv, SecretHitlerConfig
from sdb.agents.llm.openrouter_agent import OpenRouterAgent
from sdb.logging.game_logger import GameLogger


def main():
    """Run Secret Hitler game with LLM agents."""
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    print("🎮 Secret Hitler")
    print("=" * 70)
    
    # Load configuration from YAML (resolve path relative to project root)
    script_dir = Path(__file__).parent.parent  # Go up from examples/ to project root
    config_path = script_dir / "configs" / "secret_hitler.yaml"
    
    if not config_path.exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("   Using default configuration")
        config = SecretHitlerConfig(n_players=5)
    else:
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
        
        # Extract game-specific config (not agent/logging settings)
        game_config = {k: v for k, v in yaml_config.items() 
                      if k not in ['game', 'agent', 'logging']}
        config = SecretHitlerConfig(**game_config)
        print(f"✅ Loaded config from {config_path}")
    
    print(f"\n⚙️  Configuration:")
    print(f"  • Players: {config.n_players}")
    print(f"  • Features: Veto power, discussion phase, presidential powers")
    
    # Create agents (using config if available, otherwise defaults)
    print(f"\n🤖 Creating {config.n_players} LLM agents...")
    
    # Get agent settings from YAML or use defaults
    if config_path.exists():
        agent_config = yaml_config.get('agent', {})
        model = agent_config.get('model', 'anthropic/claude-3.5-sonnet')
        temperature = agent_config.get('temperature', 0.8)
        memory_capacity = agent_config.get('memory_capacity', 30)
    else:
        model = 'anthropic/claude-3.5-sonnet'
        temperature = 0.8
        memory_capacity = 30
    
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
    
    # Setup logging (using config if available)
    if config_path.exists():
        logging_config = yaml_config.get('logging', {})
        output_dir = Path(logging_config.get('output_dir', 'experiments/secret_hitler_example'))
    else:
        output_dir = Path("experiments/secret_hitler_example")
    
    logger = GameLogger(output_dir=output_dir)
    print(f"\n📁 Logs will be saved to: {output_dir}/")
    
    # Create environment
    env = SecretHitlerEnv(agents=agents, config=config, logger=logger)
    
    # Reset and show initial state
    obs = env.reset()
    print(f"\n🎲 Game Started!")
    print(f"\n👥 Roles:")
    for i, player in enumerate(env.state.players):
        role_emoji = "🔴" if player.party.value == "fascist" else "🔵"
        role_name = "🎩 HITLER" if player.is_hitler else player.role.value
        print(f"  {role_emoji} Player {i}: {role_name}")
    
    # Run game using built-in game loop
    print(f"\n▶️  Starting gameplay...")
    print("   (This may take a few minutes with LLM agents)")
    print("-" * 70)
    
    # Secret Hitler has a built-in play_game() method that handles everything
    result = env.play_game()
    
    # Show results
    print("\n" + "=" * 70)
    print("🏁 GAME OVER!")
    print("=" * 70)
    
    print(f"\n🏆 Winner: {result.winner}")
    print(f"📝 Win Reason: {result.metadata.get('win_reason', 'Game completed')}")
    
    print(f"\n📊 Final Stats:")
    print(f"  • Total rounds: {result.num_rounds}")
    print(f"  • Liberal policies: {env.state.liberal_policies}")
    print(f"  • Fascist policies: {env.state.fascist_policies}")
    print(f"  • Players alive: {len(env.state.alive_players)}/{env.num_players}")
    
    print(f"\n📁 Game log: {output_dir}/{logger.game_id}.jsonl")
    print(f"  ({len(logger.entries)} events logged)")
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    main()

