import torch

class ConnectFour:
    def __init__(self):
        self.board = [[0 for _ in range(7)] for _ in range(6)]
        self.player = 1  # 1 for player 1, -1 for player 2

    def available_moves(self):
        return [col for col, top in enumerate(self.board[0]) if top == 0]

    def make_move(self, col):
        if col not in self.available_moves():
            return False  # Indicate invalid move
        
        for row in reversed(self.board):
            if row[col] == 0:
                row[col] = self.player
                break
                
        win = self.check_win()
        self.player *= -1  # Switch players
        return win or self.is_draw()

    def check_win(self):
        for row in range(6):
            for col in range(7):
                if self.board[row][col] != 0 and self.is_part_of_win(row, col):
                    return True
        return False

    def is_part_of_win(self, row, col):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 0
            for i in range(-3, 4):
                if 0 <= row + dr * i < 6 and 0 <= col + dc * i < 7 and self.board[row + dr * i][col + dc * i] == self.board[row][col]:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
        return False

    def is_draw(self):
        return all(cell != 0 for row in self.board for cell in row)

    def is_inside_and_matches(self, player, col, row_delta):
        row_idx = sum(1 for row in self.board if row[col] != 0) - 1 + row_delta
        return 0 <= row_idx < 6 and 0 <= col < 7 and self.board[row_idx][col] == player

    def get_board(self):
        return self.board

    def reset(self):
        self.__init__()

    def is_finished(self):
        return self.check_win() or self.is_draw()

    def to_tensor(self):
        return torch.tensor(self.board).float().unsqueeze(0).float()

    def get_outcome_for_player(self, player):
        if self.check_win():
            if self.player == -player:  # The other player has made the winning move in the previous turn
                return +1
            else:
                return -1
        elif self.is_draw():
            return 0
        else:
            return None  # Game hasn't finished yet
        
    def __repr__(self):
        return str(self.board).replace("], ", "]")