class Connect4Agent:
    def __init__(self, depth):
        self.depth = depth

    def minimax(self, observation, depth, alpha, beta, maximizing):
        if depth == 0 or self.is_terminal(observation):
            return self.evaluate(observation)

        valid_moves = self.get_valid_moves(observation)

        if maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                new_board = self.make_move(observation, move, 1)  # 1 represents the agent
                eval = self.minimax(new_board, depth-1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_board = self.make_move(observation, move, 2)  # 2 represents the opponent
                eval = self.minimax(new_board, depth-1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move(self, observation):
        best_move = None
        best_value = float('-inf')
        valid_moves = self.get_valid_moves(observation)

        for move in valid_moves:
            new_board = self.make_move(observation.board, move, 1)  # 1 represents the agent
            move_value = self.minimax(new_board, self.depth-1, float('-inf'), float('inf'), False)
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
                
        return best_move

    def is_terminal(self, board):
        rows = 6
        cols = 7

        # Function to get the value at a given position
        def get_val(i, j):
            return board[i * cols + j]

        # Check for four in a row in a given direction
        def check_direction(i, j, di, dj):
            try:
                vals = [get_val(i + k*di, j + k*dj) for k in range(4)]
                if all(v == 1 for v in vals) or all(v == 2 for v in vals):
                    return True
                return False
            except IndexError:
                return False

        for i in range(rows):
            for j in range(cols):
                if check_direction(i, j, 1, 0):    # Vertical
                    return True
                if check_direction(i, j, 0, 1):    # Horizontal
                    return True
                if check_direction(i, j, 1, 1):    # Diagonal (top-left to bottom-right)
                    return True
                if check_direction(i, j, 1, -1):   # Diagonal (top-right to bottom-left)
                    return True

        # Check for draw (if the board is full)
        if 0 not in board:
            return True

        return False

    def evaluate(self, board):
        rows = 6
        cols = 7

        # Function to get the value at a given position
        def get_val(i, j):
            return board[i * cols + j]

        # Check for patterns in a given direction
        def check_direction(i, j, di, dj):
            try:
                vals = [get_val(i + k*di, j + k*dj) for k in range(4)]
                if all(v == 1 for v in vals):
                    return 1000
                if all(v == 1 for v in vals[:3]) and vals[3] == 0:
                    return 100
                if all(v == 1 for v in vals[:2]) and all(v == 0 for v in vals[2:]):
                    return 10
                if vals[0] == 1 and all(v == 0 for v in vals[1:]):
                    return 5

                if all(v == 2 for v in vals):
                    return -1000
                if all(v == 2 for v in vals[:3]) and vals[3] == 0:
                    return -100
                if all(v == 2 for v in vals[:2]) and all(v == 0 for v in vals[2:]):
                    return -10
                if vals[0] == 2 and all(v == 0 for v in vals[1:]):
                    return -5
                return 0
            except IndexError:
                return 0

        score = 0
        for i in range(rows):
            for j in range(cols):
                score += check_direction(i, j, 1, 0)    # Vertical
                score += check_direction(i, j, 0, 1)    # Horizontal
                score += check_direction(i, j, 1, 1)    # Diagonal (top-left to bottom-right)
                score += check_direction(i, j, 1, -1)   # Diagonal (top-right to bottom-left)

        return score

    def get_valid_moves(self, board):
        cols = 7
        valid_moves = []
        
        if type(board) != list:
            board = board["board"]


        for col in range(cols):
            if board[col] == 0:  # Check the topmost cell of each column
                valid_moves.append(col)
        return valid_moves
        

    

    def make_move(self, board, column, player):
        rows = 6
        cols = 7
        new_board = board.copy()

        for i in range(rows-1, -1, -1):
            if new_board[i * cols + column] == 0:
                new_board[i * cols + column] = player
                return new_board

        return new_board

agent = Connect4Agent(depth=5)
def my_agent(observation, configuration):
    return agent.get_best_move(observation)