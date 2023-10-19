import torch
import torch.nn as nn
import torch.nn.functional as F
from env import create_env
from my_agent import my_agent


class ConnectXNet(nn.Module):
    def __init__(self):
        super(ConnectXNet, self).__init__()
        self.fc1 = nn.Linear(42, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 7)  # assuming 7 columns for Connect4

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x) 
        return x

    def save(self):
        torch.save(self.state_dict(), 'model.pth')

    def test():
        pass

def train(env, model):
    wins = 0
    losses = 0
    draws = 0
    
    trainer = env.train([None, "random"])
    epochs = 400
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.SmoothL1Loss()
    initial_epsilon = 0.9
    decay_rate = 0.001
    epsilon_min = 0.1
    epsilon = initial_epsilon

    def wrapped_agent(obs, config):
        return my_agent(obs, config, model, epsilon=epsilon)

    for epoch in range(epochs):
        observation = trainer.reset()
        while not env.done:
            my_action = wrapped_agent(observation, env.configuration)
            observation, reward, done, info = trainer.step(my_action)
        
        reward = torch.tensor(reward, dtype=torch.float)
        optimizer.zero_grad()
        loss = criterion(model(torch.FloatTensor(observation.board)), reward)
        loss.backward()
        optimizer.step()

        # update the epsilon
        epsilon = max(epsilon - decay_rate, epsilon_min)


        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1

        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, wins: {wins}, losses: {losses}, draws: {draws} wins_pct: {wins / (wins + losses + draws)}')



        


def main():
    env = create_env()

    model = ConnectXNet()
    train(env, model)
    model.save()





if __name__ == "__main__":
    main()