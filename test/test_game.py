# tests/test_poker32.py
import src.poker32 as poker


def test_rewards():
    """Check reward logic for a few terminal nodes."""
    poker.init()

    # Quick sanity checks
    assert poker.rewards('cRRf', holes=["A", "K"]) == (4, -4)
    assert poker.rewards('RRc', holes=["A", "2"]) == (8, -8)
    assert poker.rewards('cc', holes=["K", "A"]) == (-2, 2)
    assert poker.rewards('cc', holes=["K", "K"]) == (0, 0)
    assert poker.rewards('cRf', holes=["K", "K"]) == (-2, 2)

    print('test_rewards passed!')


def test_game_initialisation():
    """Ensure the game object starts in the expected state."""
    game = poker.Poker32()
    assert game.button == 0
    assert game.hole_cards == ["A", "6"]
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

    for k, v in poker.GAME_TREE.items():
        print(k, v)



def test_game():
    """Ensure the game object starts in the expected state."""
    game = poker.Poker32()
    state = game._get_subjective_state(0)
    print(state)
    print(game.get_legal_moves())
    game.make_move('f')
    assert game._get_subjective_state(0)['branch'] == 'f'
    assert game.is_game_over() is True
    assert game.get_relative_rewards() == (-1, 1)

    print('test_game_initialisation passed!')


if __name__ == "__main__":
    # test_rewards()
    # test_game_initialisation()
    # test_degrees_of_freedom()
    test_game_tree()
    # test_game()
