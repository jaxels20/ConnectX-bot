import random


def my_agent(observation, configuration, model, epsilon=0):
    num_cols = configuration.columns
    board = observation.board

    import torch

    # Convert board to tensor
    board_tensor = torch.FloatTensor(board)
    board_tensor = (board_tensor - board_tensor.mean()) / board_tensor.std()

    # Predict Q-values for each column
    with torch.no_grad():
        q_values = model(board_tensor)


    # Epsilon-greedy action selection
    if random.random() < epsilon:
        # Choose a random action with probability epsilon
        available_cols = [col for col in range(num_cols) if board[col] == 0]
        return random.choice(available_cols)
    else:
        # Choose the column with the highest Q-value with probability (1 - epsilon)
        for col in q_values.argsort(descending=True):
            if board[col] == 0:
                return col.item()
    