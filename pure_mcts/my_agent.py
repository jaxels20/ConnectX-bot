from mcts  import MCTS
from ConnectState import ConnectState

from kaggle_environments import evaluate, make, utils

def find_moved_column(old_board, new_board):
    # Check if the length of both boards is the same
    if len(old_board) != len(new_board):
        raise ValueError("Both board states must have the same length.")

    # Find the position of the move by comparing the two boards
    for i, (old_val, new_val) in enumerate(zip(old_board, new_board)):
        if old_val != new_val:
            # Calculate the column based on the position of the change
            column = i % 7
            return column
    # If no move was found, return None
    return None

def make_move(board, column, player):
    """
    Makes a move on the given Connect 4 board.
    
    :param board: List[int], the current state of the board.
    :param column: int, the column in which the player wants to place their piece (0-indexed).
    :param player: int, the player number (either 1 or 2).
    
    :return: List[int], the new state of the board after making the move.
    """
    # Check if the given player number is valid
    if player not in [1, 2]:
        raise ValueError("Player number must be either 1 or 2.")

    # Check if the given column number is valid
    if column < 0 or column > 6:
        raise ValueError("Column number must be between 0 and 6 (inclusive).")
    
    # Iterate from the bottom of the column to find an empty spot
    for row in range(5, -1, -1):
        index = row * 7 + column
        if board[index] == 0:
            board[index] = player
            return board
    
    # If the column is full, raise an error
    raise ValueError(f"Column {column + 1} is already full.")


previous_board = [0] * 42
connect4board = ConnectState()
mcts = MCTS(connect4board)

def my_agent(observation, configuration):
    global previous_board

    
    board = observation.board

    # Find the last move made by the opponent
    opponent_move = find_moved_column(previous_board, board)

    # If the opponent moved first, update the ConnectState
    if opponent_move is not None:
        connect4board.move(opponent_move)
        mcts.move(opponent_move)


    mcts.search(2)
    num_rollouts, run_time = mcts.statistics()
    print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
    my_move = mcts.best_move()

    print("MCTS chose move: ", my_move)

    connect4board.move(my_move)
    mcts.move(my_move)

    previous_board = make_move(board, my_move, observation.mark)
    return my_move