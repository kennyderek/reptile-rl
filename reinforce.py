
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

        self.input_size = 4
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
    # print(env)

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
    T = 20 # number of steps per episode
    alpha = .0001 # learning rate suggested in Sutton and Barto is 1e-13
    cumulative_rewards = []
    optimizer = optim.Adam(policy.parameters(), lr=1e-5)
    lam = 0.9

    for i in tqdm(range(num_episodes)):
        S, A, R = generate_episode(policy, T)
        cumulative_rewards.append(sum(R))

        if i % 100 == 0:
            print("Average total reward:", sum(cumulative_rewards[-10:])/ 10)
            if i % 1000 == 0:
                # print(S)
                # print(A)
                pass

        for t in range(len(S)):
            G = sum([R[k] * lam**(k - t - 1) for k in range(t+1, len(S))])

            # delta_Theta = alpha * r * d/dTheta[log(p(a| theta, state_t))]
            # https://pytorch.org/docs/stable/distributions.html
            action_probs = policy(Variable(S[t]))
            m = Categorical(action_probs)
            loss = -m.log_prob(A[t]) * R[t] * alpha * lam**t
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return cumulative_rewards

if __name__ == "__main__":
    num_episodes = 2000
    policy = Policy()
    cumulative_rewards = reinforce(policy, num_episodes)

    running_average = [sum(cumulative_rewards[max(0, i-5):i])/5 for i in range(len(cumulative_rewards))]

    plt.plot(list(range(num_episodes)), running_average)
    plt.show()
