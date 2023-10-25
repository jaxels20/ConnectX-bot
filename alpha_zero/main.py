import numpy as np
from Connect4 import ConnectFour
from copy import deepcopy
from mcts import Node, mcts_search
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ConnectFourNet
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def play_game(model):
    game = ConnectFour()  # Assuming you have a ConnectFour class to manage the game state
    states, mcts_policies = [], []

    while not game.is_finished():
        root = Node(game)

        mcts_search(root, model, iterations=10000)  # Assuming 1600 MCTS iterations

        # Store game state and MCTS-derived policy
        states.append(game.get_board())
        mcts_policy = [root.children[action].visits if action in root.children else 0 for action in range(7)]
        mcts_policy = np.array(mcts_policy)

        if mcts_policy.sum() == 0:
            # If all visits are 0, make all visits equal but only for untried actions
            mcts_policy = np.array([1 if action in root.untried_actions else 0 for action in range(7)])
        mcts_policy = mcts_policy / mcts_policy.sum()  # Normalize to get probabilities
        mcts_policies.append(mcts_policy)

        # Make a move in the game. You might want to add noise or use a temperature to make exploration more diverse
        action = np.argmax(mcts_policy)
        game.make_move(action)

    outcomes = [game.get_outcome_for_player(1)] * len(states)  # Assuming get_outcome_for_player returns +1, -1, or 0

    return states, mcts_policies, outcomes

def train_network(model, states, mcts_policies, outcomes, epochs=10):
    print("Training network...")
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    policy_criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for state, mcts_policy, outcome in zip(states, mcts_policies, outcomes):
            optimizer.zero_grad()

            # Convert to PyTorch tensors
            state_tensor = torch.tensor(state).float().unsqueeze(0)
            mcts_policy_tensor = torch.tensor(mcts_policy).float().unsqueeze(0)
            outcome_tensor = torch.tensor([outcome]).float().unsqueeze(1)

            predicted_policies, predicted_value = model(state_tensor)

            # Calculate loss
            policy_loss = policy_criterion(predicted_policies, torch.argmax(mcts_policy_tensor, dim=1))
            value_loss = F.mse_loss(predicted_value, outcome_tensor)
            total_loss = policy_loss + value_loss

            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} completed")



def load_training_data(filename):
    with open(filename, 'rb') as file:
        states, actions, outcomes = pickle.load(file)
    return states, actions, outcomes

def save_training_data(filename, states, actions, outcomes):
    with open(filename, 'wb') as file:
        pickle.dump((states, actions, outcomes), file)


def main():
    model = ConnectFourNet()
    save_model = True


    # Main loop for iterative self-play and training
    model = ConnectFourNet()
    num_games_per_iteration = 5
    num_iterations = 10

    for iteration in tqdm(range(num_iterations)):
        states, mcts_policies, outcomes = [], [], []

        for _ in range(num_games_per_iteration):
            print("Playing game number", _ + 1)
            game_states, game_policies, game_outcomes = play_game(model)
            states.extend(game_states)
            mcts_policies.extend(game_policies)
            outcomes.extend(game_outcomes)

        train_network(model, states, mcts_policies, outcomes, epochs=10)

    if save_model:
        model.save()
        model.serialize()

if __name__ == "__main__":
    main()