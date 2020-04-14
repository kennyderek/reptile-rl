
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from sim import MazeSimulator, ShortCorridor

import matplotlib.pyplot as plt
from tqdm import tqdm

INPUT_SIZE = 1


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        self.input_size = INPUT_SIZE
        self.hidden_size = 100

        # self.ACTION_SPACE = {0: "N", 1: "S", 2: "E", 3: "W"}
        self.ACTION_SPACE = {0: "R", 1: "L"}

        self.num_actions = len(self.ACTION_SPACE.keys())

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)

        self.fc5 = nn.Linear(self.hidden_size, self.num_actions)
        self.softmax = nn.Softmax()


    def forward(self, x):
        '''
        x: input vector describing state
        return: vector containing probabilities?? of each
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))

        x = self.fc5(x)
        return self.softmax(x)

def generate_episode(policy, T):
    env = ShortCorridor()
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

def compute_policy_loss(policy, state, action, weights):
    '''
    weights is what to multiply the log probability by
    '''
    logp = Categorical(policy(state)).log_prob(action)
    return -(logp * weights).mean()

def reinforce(policy, num_episodes, lr):
    T = 90 # number of steps per episode
    alpha_p = 1 # learning rate suggested in Sutton and Barto is 2**-13, we instead use Adam with an lr argument
    cumulative_rewards = []
    opt_p = optim.Adam(policy.parameters(), lr=lr)
    lam = 0.9

    for i in tqdm(range(num_episodes)):

        S, A, R = generate_episode(policy, T)
        cumulative_rewards.append(sum(R))

        if i % 100 == 0:
            print("Average total reward:", sum(cumulative_rewards[-10:])/ 10)

        batch_states = []
        batch_actions = []
        batch_weights = []
        batch_rewards = []
        I = 1
        losses_v = []
        losses_p = []
        for t in range(len(S) - 1):
            G = sum([R[k] * lam**(k - t - 1) for k in range(t+1, len(S))]) * lam**t * alpha_p

            batch_states.append(S[t].data.numpy())
            batch_actions.append(A[t])
            batch_weights.append(G)

        batch_policy_loss = compute_policy_loss(policy=policy,
                                state=torch.as_tensor(batch_states, dtype=torch.float32),
                                action=torch.as_tensor(batch_actions, dtype=torch.float32),
                                weights=torch.as_tensor(batch_weights, dtype=torch.float32))

        # delta_Theta = alpha * r * d/dTheta[log(p(a| theta, state_t))]
        # https://pytorch.org/docs/stable/distributions.html
        opt_p.zero_grad()
        batch_policy_loss.backward(retain_graph=True)
        opt_p.step()

    return cumulative_rewards

if __name__ == "__main__":

    num_episodes = 4000
    num_runs = 10

    for lr in [0.001]:
        all_runs = []
        all_action_probs = []

        for run_iter in range(0, num_runs):
            print("Run number:", run_iter)
            policy = Policy()
            cumulative_rewards = reinforce(policy, num_episodes, lr)

            running_average = [sum(cumulative_rewards[max(0, i-50):i])/50 for i in range(len(cumulative_rewards))]

            all_runs.append(running_average)

        data = np.mean(np.array(all_runs), axis=0)

        for run in all_runs:
            plt.plot(list(range(num_episodes)), run)
        plt.savefig("%s.png" % (lr))
        plt.clf()

