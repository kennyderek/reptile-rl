
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

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        self.input_size = STATE_INPUT_SIZE
        self.hidden_size = 100

        self.ACTION_SPACE = ACTION_SPACE

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

def generate_episode(policy, env, T):
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

def reinforce(policy, lr):
    alpha_p = 1 # learning rate suggested in Sutton and Barto is 2**-13, we instead use Adam with an lr argument
    cumulative_rewards = []
    opt_p = optim.Adam(policy.parameters(), lr=lr)
    lam = 0.9

    for i in tqdm(range(NUM_EPOCHS+1)):

        if i % 200 == 0 and i > 0:
            WORLD().visualize(policy, i)
            print("Average total reward:", sum(cumulative_rewards[-10:])/min(10, len(cumulative_rewards)))

        cumulative_rewards_batch = []
        batch_states = []
        batch_actions = []
        batch_weights = []
        batch_rewards = []

        # should an epoch be within a single world?  
        env = WORLD()
        for _ in range(NUM_EPISODES_PER_EPOCH):
            env = WORLD()
            S, A, R = generate_episode(policy, env, MAX_EPISODE_LEN)
            cumulative_rewards_batch.append(sum(R))

            for t in range(len(S) - 1):
                G = sum([R[k] * lam**(k - t - 1) for k in range(t+1, len(S))]) * lam**t * alpha_p

                batch_states.append(S[t].data.numpy())
                batch_actions.append(A[t])
                batch_weights.append(G)

            # Do we need to do this? What if we generate a new env within a batch gradient update?
            env.reset_soft()

        batch_policy_loss = compute_policy_loss(policy=policy,
                                state=torch.as_tensor(batch_states, dtype=torch.float32),
                                action=torch.as_tensor(batch_actions, dtype=torch.float32),
                                weights=torch.as_tensor(batch_weights, dtype=torch.float32))

        # delta_Theta = alpha * r * d/dTheta[log(p(a| theta, state_t))]
        # https://pytorch.org/docs/stable/distributions.html
        opt_p.zero_grad()
        batch_policy_loss.backward(retain_graph=True)
        opt_p.step()

        cumulative_rewards.append(sum(cumulative_rewards_batch)/NUM_EPISODES_PER_EPOCH)

    return cumulative_rewards

# def visualize_policy(policy):

def train_one_run(learning_rate):
    '''
    learning rate: the rate at which to train this policy

    returns: the 50 episode moving average of reward across this run
    '''
    policy = Policy()
    cumulative_rewards = reinforce(policy, learning_rate)
    running_average = [sum(cumulative_rewards[max(0, i-50):i])/min(50, i+1) for i in range(len(cumulative_rewards))]
    return running_average


if __name__ == "__main__":
    '''
    parameters to be changed if the environment changes
    '''
    STATE_INPUT_SIZE = 2
    WORLD = [ShortCorridor, MazeSimulator][1]
    ACTION_SPACE = {ShortCorridor: {0: "R", 1: "L"}, MazeSimulator: {0: "N", 1: "S", 2: "E", 3: "W"}}[WORLD]

    '''
    parameters for the training
    '''
    VISUALIZE_POLICY = True
    NUM_RUNS = 1 # number of times to train a policy from scratch (to handle bad random seed, etc.)
    NUM_EPOCHS = 2000 # an epoch is a complete walkthrough of NUM_EPISODES_PER_EPOCH games
    NUM_EPISODES_PER_EPOCH = 5
    MAX_EPISODE_LEN = 100

    for learning_rate in [1e-3]:
        all_runs = []

        for run_iter in range(0, NUM_RUNS):
            print("Starting run number:", run_iter)
            all_runs.append(train_one_run(learning_rate))

        data = np.mean(np.array(all_runs), axis=0)

        for run in all_runs:
            plt.plot(list(range(len(run))), run)
        plt.savefig("LearningRate%s.png" % (learning_rate))
        plt.clf()

