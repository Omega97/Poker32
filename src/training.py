import random
from src.rl_agent import AgentRL
from src.poker32 import Poker32


def training(file_path,
             config,
             name='training_agent',
             rng: random.Random | None = None):

    # Load the agent with training mode enabled
    if file_path.exists():
        print(f"Loading existing policy from {file_path}")
        agent = AgentRL.load(str(file_path),
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
    print(f"Target: {config['n_epochs'] * config['n_cycles']:,} games\n")

    # Training loop
    for cycle in range(config["n_cycles"]):
        for _ in range(config["n_epochs"]):
            game.play((agent, agent))

        print(f"\rCycle {cycle+1}/{config['n_cycles']} | "
              f"Games played: {agent.games_played:,}", end='')

    print()
    agent.save(file_path)
    print(f"Training complete! Policy saved.")
