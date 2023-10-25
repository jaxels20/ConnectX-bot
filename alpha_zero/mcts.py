import numpy as np

class Node:
    def __init__(self, game_state, parent=None, prior=0):
        self.game_state = game_state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0  # Rename this from "value" to avoid confusion
        self.prior = prior  # Probability given by the neural network policy
        self.untried_actions = game_state.available_moves()

    def select(self, c_puct=1.0):
        # Select a child node according to the UCB formula with neural network prior
        s = sum(child.visits for child in self.children.values())
        return max(self.children.values(), key=lambda c: (c.value_sum / (c.visits + 1e-7) + c_puct * self.prior * np.sqrt(s) / (1 + c.visits)))

    def expand(self, action_priors, add_noise=True, alpha=1.0, epsilon=0.25):
        if add_noise:
            noise = np.random.dirichlet([alpha] * len(action_priors))
            for i, (action, prob) in enumerate(action_priors):
                action_priors[i] = (action, (1-epsilon)*prob + epsilon*noise[i])

        for action, prob in action_priors:
            if action not in self.children:
                next_game_state = deepcopy(self.game_state)
                next_game_state.make_move(action)
                self.children[action] = Node(next_game_state, parent=self, prior=prob)


    def backpropagate(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(-value)

    def is_fully_expanded(self):
        return len(self.children) == len(self.untried_actions)

    def is_terminal(self):
        return self.game_state.check_win() or self.game_state.is_draw()

    def __repr__(self) -> str:
        
        return_str = f"Node(game_state={self.game_state}, \nvisits={self.visits}, \nvalue_sum={self.value_sum}, \nprior={self.prior}, \nuntried_actions={self.untried_actions}, \nis_fully_expanded={self.is_fully_expanded()}, \nis_terminal={self.is_terminal()})"
        return return_str

from copy import deepcopy

def simulate_random_game(game_state):
    """
    Simulate a random game from the given state and return the outcome from the perspective
    of the current player in the game state:
    +1 if they win, -1 if they lose, and 0 for a draw.
    """
    while not game_state.is_finished():
        moves = game_state.available_moves()
        move = np.random.choice(moves)
        outcome = game_state.make_move(move)

        # If the move resulted in a win or draw
        if outcome:
            # Return outcome from the perspective of the original player
            return game_state.get_outcome_for_player(game_state.player)
    
    # This point shouldn't be reached, but return None for safety
    return None


def mcts_search(root, model, iterations=1000):
    for _ in range(iterations):
        node = root
        # Selection
        num_nodes_expanded = 0
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select(c_puct=3.0)
            num_nodes_expanded += 1
    
        # Expansion
        if not node.is_terminal():
            policy, value = model(node.game_state.to_tensor())
            policy_array = policy.squeeze().detach().cpu().numpy()
            legal_actions = node.game_state.available_moves()  # Make sure available_moves returns a list of legal columns
            action_priors = [(action, policy_array[action]) for action in legal_actions]
            node.expand(action_priors, add_noise=True)
            node.backpropagate(value.item())  # Use value.item() to get the scalar value

        # Simulation (can be skipped if relying solely on neural network value)
            """ result = simulate_random_game(deepcopy(node.game_state))
            node.backpropagate(result) """

    # Return the column (action) with the most visits
    return max(root.children.keys(), key=lambda action: root.children[action].visits)