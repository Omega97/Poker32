import random
import pathlib
from src.agents.rl_agent import AgentRL
from src.poker32 import Poker32


def training(agent_class,
             file_path: pathlib.Path,
             config: dict,
             name='training_agent',
             rng: random.Random | None = None):

    # Load the agent with training mode enabled
    if file_path.exists():
        print(f"Loading existing policy from {file_path}")
        agent = agent_class.load(str(file_path),
                                 config=config,
                                 name=name,
                                 training=True)
        print(f"Resuming from {agent.games_played:,} games")
    else:
        print("No existing policy found → starting from scratch")
        agent = AgentRL(config=config,
                        name=name,
                        rng=rng,
                        training=True)

    # Game instance
    game = Poker32(rng=rng)
    print("Starting Poker32 RL training (additive logit, T=1.0)")
    print(f"Target: {config['batch_size'] * config['n_cycles']:,} games\n")

    # Training loop
    for cycle in range(config["n_cycles"]):
        for _ in range(config["batch_size"]):
            game.play((agent, agent))

        maturity = agent.get_maturity()
        print(f"\rCycle {cycle+1}/{config['n_cycles']} | "
              f"Games played: {agent.games_played:,} | "
              f"mat={maturity:.2%}", end='')

    print()
    agent.save(file_path)
    print(f"Training complete! Policy saved.")
