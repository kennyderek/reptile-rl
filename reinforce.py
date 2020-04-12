
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from sim import WorldSimulator

import matplotlib.pyplot as plt
from tqdm import tqdm

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        self.input_size = 6
        self.hidden_size = 64
        self.num_actions = 4

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_actions)
        self.softmax = nn.Softmax()

        self.ACTION_SPACE = {0: "N", 1: "S", 2: "E", 3: "W"}

    def forward(self, x):
        '''
        x: input vector describing state
        return: vector containing probabilities?? of each
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.softmax(x)

def generate_episode(policy, T):
    env = WorldSimulator()

    S, A, R = [], [], []
    for i in range(0, T):
        state = Variable(torch.FloatTensor(env.get_state()))
        action_probs = policy(state)
        m = Categorical(action_probs)
        action_idx = m.sample()
        action = policy.ACTION_SPACE[action_idx.item()]
        next_state, reward = env.step(action)

        S.append(state)
        A.append(action_idx)
        R.append(reward)

        if next_state == None:
            # reached terminal state
            break
        else:
            state = next_state
    
    return S, A, R


def reinforce(policy, num_episodes):
    T = 100 # number of steps per episode
    alpha = .001
    cumulative_rewards = []
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    for _ in tqdm(range(num_episodes)):
        S, A, R = generate_episode(policy, T)
        cumulative_rewards.append(sum(R))

        for t in range(len(S)):
            G = sum(R[t+1:]) # does not include discount rate lambda

            # delta_Theta = alpha * r * d/dTheta[log(p(a| theta, state_t))]
            # https://pytorch.org/docs/stable/distributions.html
            action_probs = policy(Variable(S[t]))
            m = Categorical(action_probs)
            loss = -m.log_prob(A[t]) * R[t] * alpha
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return cumulative_rewards

if __name__ == "__main__":
    num_episodes = 500
    policy = Policy()
    cumulative_rewards = reinforce(policy, num_episodes)

    plt.plot(list(range(num_episodes)), cumulative_rewards)
    plt.show()
