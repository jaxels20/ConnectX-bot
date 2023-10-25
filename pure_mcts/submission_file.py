from copy import deepcopy
import numpy as np

import math

class GameMeta:
    PLAYERS = {'none': 0, 'one': 1, 'two': 2}
    OUTCOMES = {'none': 0, 'one': 1, 'two': 2, 'draw': 3}
    INF = float('inf')
    ROWS = 6
    COLS = 7


class MCTSMeta:
    EXPLORATION = math.sqrt(2)



class ConnectState:
    def __init__(self):
        self.board = [[0] * GameMeta.COLS for _ in range(GameMeta.ROWS)]
        self.to_play = GameMeta.PLAYERS['one']
        self.height = [GameMeta.ROWS - 1] * GameMeta.COLS
        self.last_played = []

    def get_board(self):
        return deepcopy(self.board)

    def move(self, col):
        self.board[self.height[col]][col] = self.to_play
        self.last_played = [self.height[col], col]
        self.height[col] -= 1
        self.to_play = GameMeta.PLAYERS['two'] if self.to_play == GameMeta.PLAYERS['one'] else GameMeta.PLAYERS['one']

    def get_legal_moves(self):
        return [col for col in range(GameMeta.COLS) if self.board[0][col] == 0]

    def check_win(self):
        if len(self.last_played) > 0 and self.check_win_from(self.last_played[0], self.last_played[1]):
            return self.board[self.last_played[0]][self.last_played[1]]
        return 0

    def check_win_from(self, row, col):
        player = self.board[row][col]
        """
        Last played action is at (row, col)
        Check surrounding 7x7 grid for a win
        """

        consecutive = 1
        # Check horizontal
        tmprow = row
        while tmprow + 1 < GameMeta.ROWS and self.board[tmprow + 1][col] == player:
            consecutive += 1
            tmprow += 1
        tmprow = row
        while tmprow - 1 >= 0 and self.board[tmprow - 1][col] == player:
            consecutive += 1
            tmprow -= 1

        if consecutive >= 4:
            return True

        # Check vertical
        consecutive = 1
        tmpcol = col
        while tmpcol + 1 < GameMeta.COLS and self.board[row][tmpcol + 1] == player:
            consecutive += 1
            tmpcol += 1
        tmpcol = col
        while tmpcol - 1 >= 0 and self.board[row][tmpcol - 1] == player:
            consecutive += 1
            tmpcol -= 1

        if consecutive >= 4:
            return True

        # Check diagonal
        consecutive = 1
        tmprow = row
        tmpcol = col
        while tmprow + 1 < GameMeta.ROWS and tmpcol + 1 < GameMeta.COLS and self.board[tmprow + 1][tmpcol + 1] == player:
            consecutive += 1
            tmprow += 1
            tmpcol += 1
        tmprow = row
        tmpcol = col
        while tmprow - 1 >= 0 and tmpcol - 1 >= 0 and self.board[tmprow - 1][tmpcol - 1] == player:
            consecutive += 1
            tmprow -= 1
            tmpcol -= 1

        if consecutive >= 4:
            return True

        # Check anti-diagonal
        consecutive = 1
        tmprow = row
        tmpcol = col
        while tmprow + 1 < GameMeta.ROWS and tmpcol - 1 >= 0 and self.board[tmprow + 1][tmpcol - 1] == player:
            consecutive += 1
            tmprow += 1
            tmpcol -= 1
        tmprow = row
        tmpcol = col
        while tmprow - 1 >= 0 and tmpcol + 1 < GameMeta.COLS and self.board[tmprow - 1][tmpcol + 1] == player:
            consecutive += 1
            tmprow -= 1
            tmpcol += 1

        if consecutive >= 4:
            return True

        return False

    def game_over(self):
        return self.check_win() or len(self.get_legal_moves()) == 0

    def get_outcome(self):
        if len(self.get_legal_moves()) == 0 and self.check_win() == 0:
            return GameMeta.OUTCOMES['draw']

        return GameMeta.OUTCOMES['one'] if self.check_win() == GameMeta.PLAYERS['one'] else GameMeta.OUTCOMES['two']

    def print(self):
        print('=============================')

        for row in range(GameMeta.ROWS):
            for col in range(GameMeta.COLS):
                print('| {} '.format('X' if self.board[row][col] == 1 else 'O' if self.board[row][col] == 2 else ' '), end='')
            print('|')

        print('=============================')



import random
import time
import math
from copy import deepcopy


class Node:
    def __init__(self, move, parent):
        self.move = move
        self.parent = parent
        self.N = 0
        self.Q = 0
        self.children = {}
        self.outcome = GameMeta.PLAYERS['none']

    def add_children(self, children: dict) -> None:
        for child in children:
            self.children[child.move] = child

    def value(self, explore: float = MCTSMeta.EXPLORATION):
        if self.N == 0:
            return 0 if explore == 0 else GameMeta.INF
        else:
            return self.Q / self.N + explore * math.sqrt(math.log(self.parent.N) / self.N)


class MCTS:
    def __init__(self, state=ConnectState()):
        self.root_state = deepcopy(state)
        self.root = Node(None, None)
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0

    def select_node(self) -> tuple:
        node = self.root
        state = deepcopy(self.root_state)

        while len(node.children) != 0:
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value()).value()
            max_nodes = [n for n in children if n.value() == max_value]

            node = random.choice(max_nodes)
            state.move(node.move)

            if node.N == 0:
                return node, state

        if self.expand(node, state):
            node = random.choice(list(node.children.values()))
            state.move(node.move)

        return node, state

    def expand(self, parent: Node, state: ConnectState) -> bool:
        if state.game_over():
            return False

        children = [Node(move, parent) for move in state.get_legal_moves()]
        parent.add_children(children)

        return True

    def roll_out(self, state: ConnectState) -> int:
        while not state.game_over():
            state.move(random.choice(state.get_legal_moves()))

        return state.get_outcome()

    def back_propagate(self, node: Node, turn: int, outcome: int) -> None:

        # For the current player, not the next player
        reward = 0 if outcome == turn else 1

        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent
            if outcome == GameMeta.OUTCOMES['draw']:
                reward = 0
            else:
                reward = 1 - reward

    def search(self, time_limit: int):
        start_time = time.process_time()

        num_rollouts = 0
        while time.process_time() - start_time < time_limit:
            node, state = self.select_node()
            outcome = self.roll_out(state)
            self.back_propagate(node, state.to_play, outcome)
            num_rollouts += 1

        run_time = time.process_time() - start_time
        self.run_time = run_time
        self.num_rollouts = num_rollouts

    def best_move(self):
        if self.root_state.game_over():
            return -1

        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)

        return best_child.move

    def move(self, move):
        if move in self.root.children:
            self.root_state.move(move)
            self.root = self.root.children[move]
            return

        self.root_state.move(move)
        self.root = Node(None, None)

    def statistics(self) -> tuple:
        return self.num_rollouts, self.run_time


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


    mcts.search(5)
    num_rollouts, run_time = mcts.statistics()
    print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
    my_move = mcts.best_move()

    print("MCTS chose move: ", my_move)

    connect4board.move(my_move)
    mcts.move(my_move)

    previous_board = make_move(board, my_move, observation.mark)
    return my_move