# tests/test_poker32.py
import random
import src.poker32 as poker


def test_rewards():
    """Check reward logic for a few terminal nodes."""
    poker.init()

    dct = [('cRRf', {"SB": "A", "BB": "K"}, {"SB": 4, "BB": -4}),
           ('RRc', {"SB": "A", "BB": "2"}, {"SB": 8, "BB": -8}),
           ('cc', {"SB": "K", "BB": "A"}, {"SB": -2, "BB": 2}),
           ('cc', {"SB": "K", "BB": "K"}, {"SB": 0, "BB": 0}),
           ('cRf', {"SB": "K", "BB": "K"},  {"SB": -2, "BB": 2})
           ]

    # Quick sanity checks
    for leaf, hole_cards, rewards in dct:
        game_rewards = poker.rewards(leaf, hole_cards)
        if not game_rewards == rewards:
            raise ValueError(f'{game_rewards} != {rewards}')

    print('test_rewards passed!')


def test_game_initialisation():
    """Ensure the game object starts in the expected state."""
    rng = random.Random(0)
    game = poker.Poker32(rng=rng)

    assert game.button == 0
    assert game.hole_cards == ["Q", "2"]
    game._init_hole_cards(holes=["Q", "Q"])
    assert game.hole_cards == ["Q", "Q"]

    print('test_game_initialisation passed!')


def test_degrees_of_freedom():
    """Display a rough 'degrees-of-freedom' count."""
    poker.init()
    branching = sum(len(children) - 1 for children in poker.GAME_TREE.values())
    dof = branching * poker.N_RANKS
    assert dof == 793
    # print(poker.GAME_TREE)

    print('test_degrees_of_freedom passed!')


def test_game_tree():
    poker.init()
    leafs = ['', 'c', 'Q', 'RRRR']
    for leaf in leafs:
        assert leaf in poker.GAME_TREE

    # for k, v in poker.GAME_TREE.items():
    #     print(k, v)


def test_game():
    """Ensure the game object starts in the expected state."""
    game = poker.Poker32()
    # state = game._get_subjective_state(0)
    # print(state)
    # print(game.get_legal_moves())
    print(game.get_positions())

    game.make_move('f')
    assert game._get_subjective_state(0)['branch'] == 'f'
    assert game.is_game_over() is True
    assert game.get_rewards() == {"SB": -1, "BB": 1}

    print('test_game_initialisation passed!')


if __name__ == "__main__":
    test_rewards()
    test_game_initialisation()
    test_degrees_of_freedom()
    test_game_tree()
    test_game()
