from Connect4 import ConnectFour

def test_available_moves():
    game = ConnectFour()
    assert game.available_moves() == [0, 1, 2, 3, 4, 5, 6], "Error in available_moves at the start"

    game.make_move(0)
    assert game.available_moves() == [0, 1, 2, 3, 4, 5, 6], "Error in available_moves after a move"

    for _ in range(6):
        game.make_move(0)
    assert game.available_moves() == [1, 2, 3, 4, 5, 6], "Error in available_moves when column is full"


def test_check_win():
    game = ConnectFour()

    # Horizontal win
    for _ in range(3):
        game.make_move(0)
        game.make_move(1)
    game.make_move(0)
    assert game.check_win(), "Error in check_win for horizontal win"

    # Vertical win
    game = ConnectFour()
    for _ in range(4):
        game.make_move(0)
    assert game.check_win(), "Error in check_win for vertical win"

    # Diagonal (left to right) win
    game = ConnectFour()
    game.make_move(0)
    game.make_move(1)
    game.make_move(1)
    game.make_move(2)
    game.make_move(2)
    game.make_move(2)
    game.make_move(3)
    game.make_move(3)
    game.make_move(3)
    game.make_move(3)
    assert game.check_win(), "Error in check_win for diagonal (left to right) win"


def test_is_draw():
    game = ConnectFour()

    # Not a draw
    for col in range(7):
        for _ in range(3):
            game.make_move(col)
    assert not game.is_draw(), "Error in is_draw when the board is not full"

    # A draw
    for col in range(7):
        game.make_move(col)
    assert game.is_draw(), "Error in is_draw when the board is full"

def test_make_move():
    game = ConnectFour()

    # Test basic move
    game.make_move(0)
    assert game.board[5][0] == 1, "Error: make_move didn't correctly place the piece"
    assert game.player == 2, "Error: make_move didn't switch player"

    # Fill a column
    for _ in range(5):
        game.make_move(0)

    assert game.board[0][0] == 2, "Error: make_move didn't place the piece at the top of the column"

    # Overfill a column
    result = game.make_move(0)
    assert not result, "Error: make_move allowed overfilling a column"
    assert game.player == 1, "Error: make_move switched player after invalid move"

    # Check vertical win
    game = ConnectFour()
    for _ in range(4):
        game.make_move(0)
        game.make_move(1)
    assert game.check_win(), "Error in check_win for vertical win"

    # Check horizontal win
    game = ConnectFour()
    for i in range(4):
        game.make_move(i)
        game.make_move(i)
    assert game.check_win(), "Error in check_win for horizontal win"

def test_is_draw():
    game = ConnectFour()

    # Manually set the board to a full state
    game.board = [[1, 2, 1, 2, 1, 2, 1],
                  [2, 1, 2, 1, 2, 1, 2],
                  [1, 2, 1, 2, 1, 2, 1],
                  [2, 1, 2, 1, 2, 1, 2],
                  [1, 2, 1, 2, 1, 2, 1],
                  [2, 1, 2, 1, 2, 1, 2]]
    assert game.is_draw(), "Error: Game should be a draw when the board is full"

    # Now create a non-full board state
    game.board[5][6] = 0  # Empty one cell
    assert not game.is_draw(), "Error: Game shouldn't be a draw when the board is not full"

def test_available_moves():
    game = ConnectFour()
    assert set(game.available_moves()) == {0, 1, 2, 3, 4, 5, 6}, "Error: Initially, all columns should be available."

    # Fill up column 0 completely
    for _ in range(6):
        game.make_move(0)
    assert set(game.available_moves()) == {1, 2, 3, 4, 5, 6}, "Error: Column 0 should be unavailable."

def test_make_move_and_player_switch():
    game = ConnectFour()
    game.make_move(0)
    assert game.board[5][0] == 1, "Error: First move in empty column should be at bottom row."
    assert game.player == 2, "Error: Player should switch to 2 after player 1 makes a move."

    game.make_move(0)
    assert game.board[4][0] == 2, "Error: Second move in column should be one row above."
    assert game.player == 1, "Error: Player should switch back to 1 after player 2 makes a move."


def test_edge_cases():
    game = ConnectFour()
    for _ in range(6):
        game.make_move(0)
    try:
        game.make_move(0)
        assert False, "Error: Making a move in a full column should raise an error."
    except:
        pass

def test_board_logic():
    test_available_moves()
    #test_check_win()
    test_is_draw()
    test_make_move()
    test_is_draw()
    test_available_moves()
    test_make_move_and_player_switch()
    test_edge_cases()
    print("All tests passed!")


if __name__ == "__main__":
    test_board_logic()