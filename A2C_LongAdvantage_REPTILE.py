
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
from random import random
import copy

class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()

        self.input_size = STATE_INPUT_SIZE
        self.hidden_size = 400

        self.ACTION_SPACE = ACTION_SPACE

        self.num_actions = len(self.ACTION_SPACE.keys())

        self.fc1_a = nn.Linear(self.input_size, self.hidden_size)
        self.fc2_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6_a = nn.Linear(self.hidden_size, self.num_actions)
        self.softmax = nn.Softmax()

    def forward(self, x):
        '''
        x: input vector describing state
        return: vector containing probabilities?? of each
        '''
        x = F.relu(self.fc1_a(x))
        x = F.relu(self.fc2_a(x))
        x = F.relu(self.fc3_a(x))
        x = F.relu(self.fc4_a(x))
        x = F.relu(self.fc5_a(x))
        x = self.fc6_a(x)
        return self.softmax(x)

class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()

        self.input_size = STATE_INPUT_SIZE
        self.hidden_size = 400

        # Define the critic
        self.fc1_c = nn.Linear(self.input_size, self.hidden_size)
        self.fc2_c = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_c = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4_c = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5_c = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6_c = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        '''
        x is a state, return estimate of its value
        '''

        x = F.relu(self.fc1_c(x))
        x = F.relu(self.fc2_c(x))
        x = F.relu(self.fc3_c(x))
        x = F.relu(self.fc4_c(x))
        x = F.relu(self.fc5_c(x))
        return self.fc6_c(x)


def generate_episode(policy, env, T):
    S, A, R = [], [], []
    for i in range(0, T):
        state = Variable(torch.FloatTensor(env.get_state_onehot()))
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
    
    S.append(torch.FloatTensor(next_state) if next_state != None else None)
    return S, A, R

def compute_actor_loss_PPO(actor, old_actor, state, action, weights, entropy_weight):
    '''
    weights is what to multiply the log probability by
    '''
    # logp = Categorical(actor(state)).log_prob(action)
    # return -(logp * weights).mean()
    logp_ratios =  Categorical(actor(state)).log_prob(action) - Categorical(old_actor(state)).log_prob(action)
    # print(logp_ratios)
    ratios = torch.exp(logp_ratios)
    clipped_adv = torch.clamp(ratios, 1 - 0.02, 1 + 0.02) * weights
    non_clipped_adv = ratios * weights

    entropy = Categorical(actor(state)).entropy().mean()

    return -(torch.min(clipped_adv, non_clipped_adv).mean() + entropy_weight*entropy)

def compute_actor_loss(actor, state, action, weights):
    '''
    weights is what to multiply the log probability by
    '''
    logp = Categorical(actor(state)).log_prob(action)
    return -(logp * weights).mean()


def compute_critic_loss(critic, state, weights):
    '''
    weights is what to multiply the log probability by
    '''
    loss = critic(state)*weights
    return loss.mean()

def A2C(env, init_actor, init_critic, num_train_steps, lr):
    cumulative_rewards = []
    lam = 1

    # old actor should equal the policy initialization?
    actor = Actor()
    critic = Critic()
    actor.load_state_dict(copy.deepcopy(init_actor.state_dict()))
    critic.load_state_dict(copy.deepcopy(init_critic.state_dict()))
    opt_a = optim.Adam(actor.parameters(), lr=lr/3)
    opt_c = optim.Adam(critic.parameters(), lr=lr)

    # old_actor.load_state_dict(copy.deepcopy(actor.state_dict()))

    cumulative_rewards_batch = []
    current_timesteps_in_env = 0


    for i in tqdm(range(num_train_steps)):

        if i % 500 == 0 and i > 0:
            WORLD().visualize(actor, i)
            WORLD().visualize_value(critic, i)
            print("Average total reward:", sum(cumulative_rewards[-10:])/min(10, len(cumulative_rewards)))
        
        batch_states = []
        batch_actions = []
        batch_weights = []

        if current_timesteps_in_env > MAX_EPISODE_LEN:
            cumulative_rewards.append(sum(cumulative_rewards_batch))
            cumulative_rewards_batch = []
            env.reset_soft()
            current_timesteps_in_env = 0

        # should an epoch be within a single world?  
        T = 20
        current_timesteps_in_env += T
        S, A, R = generate_episode(actor, env, T) # T is 10
        cumulative_rewards_batch.extend(R)

        discounted_rewards = [0] * len(R)
        reward_tplusone = 0
        # compute discounted rewards
        for t in reversed(list(range(len(R)))):
            discounted_rewards[t] = R[t] + lam * reward_tplusone
            reward_tplusone = discounted_rewards[t]


        # compute advantage (of that action)
        adv = [0]*len(A)
        for t in range(len(A)):
            if S[t+1] == None:
                adv[t] = torch.as_tensor(np.array(R[t]), dtype=torch.float32) # reward for transitioning to final state
                cumulative_rewards.append(sum(cumulative_rewards_batch))
                cumulative_rewards_batch = []
                env.reset_soft()
            else:
                # print(S[t+1])
                # print(S[t])
                # print("advantage:", R[t] + lam*critic(S[t+1]) - critic(S[t]))
                adv[t] = discounted_rewards[t] + lam**(len(A)-t)*critic(S[len(A)-1]) - critic(S[t])

        # print(adv)
        # L_V = discounted_rewards[t] + V(s_{t+1}) - V(s_t)
        # print(adv)
        for t in range(len(A)):
            # G = sum([R[k] * lam**(k - t - 1) for k in range(t+1, len(S))]) * lam**t

            batch_states.append(S[t].data.numpy())
            batch_actions.append(A[t])
            batch_weights.append(adv[t].item())

            # Do we need to do this? What if we generate a new env within a batch gradient update?
            # if random() > 0.6:
            # env.reset_soft()
            # else:
            #     env = WORLD()

        # print("weight", batch_weights)
        std = np.std(batch_weights)
        if std != 0:
            batch_weights = np.array(batch_weights)
            batch_weights = (batch_weights - batch_weights.mean()) / std
        # print("weights:", batch_weights)

        batch_actor_loss = compute_actor_loss(actor=actor,
                                state=torch.as_tensor(batch_states, dtype=torch.float32),
                                action=torch.as_tensor(batch_actions, dtype=torch.float32),
                                weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        
        batch_critic_loss = compute_critic_loss(critic=critic,
                                state=torch.as_tensor(batch_states, dtype=torch.float32),
                                weights=torch.as_tensor(batch_weights, dtype=torch.float32))

        # old_actor.load_state_dict(copy.deepcopy(actor.state_dict()))

        # delta_Theta = alpha * r * d/dTheta[log(p(a| theta, state_t))]
        # https://pytorch.org/docs/stable/distributions.html
        opt_a.zero_grad()
        batch_actor_loss.backward()
        opt_a.step()

        opt_c.zero_grad()
        batch_critic_loss.backward()
        opt_c.step()

        # cumulative_rewards.append(sum(cumulative_rewards_batch)/NUM_EPISODES_PER_EPOCH)

    return cumulative_rewards, actor, critic

# def visualize_policy(policy):

def train_one_run(learning_rate, actor = None, critic = None, env = None):
    '''
    learning rate: the rate at which to train this policy

    returns: the 50 episode moving average of reward across this run
    '''
    if actor == None or critic == None:
        actor = Actor()
        critic = Critic()
    
    cumulative_rewards = A2C(env, actor, critic, 2000, learning_rate)
    # running_average = [sum(cumulative_rewards[max(0, i-50):i])/min(50, i+1) for i in range(len(cumulative_rewards))]
    # return running_average

def paramter_diff_loss(target, old):
    # loss = 0
    '''
    move old closer to target by setting its grad data to the difference between target and old
    '''
    for target_p, old_p in zip(target.parameters(), old.parameters()):
        if old_p.grad is None:
            # if self.is_cuda():
                # p.grad = Variable(torch.zeros(p.size())).cuda()
            # else:
            old_p.grad = Variable(torch.zeros(old_p.size()))
        # old_p.grad.data.zero_()  # not sure this is required
        old_p.grad.data.add_((target_p.data - old_p.data)/1e-4) # TODO: determine best division by
        # print((target_p.data - old_p.data).mean())
    # return loss.mean()

def train_reptile():
    init_actor = Actor()
    init_critic = Critic()
    opt_init_actor = optim.SGD(init_actor.parameters(), lr=1e-4, momentum=0)
    opt_init_critic = optim.SGD(init_critic.parameters(), lr=1e-4, momentum=0)

    for iter in tqdm(range(0, 30)):
        env = WORLD()
        print(env)

        cumulative_rewards, actor, critic = A2C(env, init_actor, init_critic, 200, 3e-4) # k = 10 train steps?

        opt_init_actor.zero_grad()
        paramter_diff_loss(actor, init_actor)
        # actor_loss = paramter_diff_loss(actor, init_actor)
        # actor_loss.backward()
        opt_init_actor.step()

        opt_init_critic.zero_grad()
        paramter_diff_loss(critic, init_critic)
        # critic_loss = paramter_diff_loss(critic, init_critic)
        # critic_loss.backward()
        opt_init_critic.step()

        env.visualize_value(init_critic, -88)
        env.visualize(init_actor, -88)
    
    return init_actor, init_critic


if __name__ == "__main__":
    '''
    parameters to be changed if the environment changes
    '''
    STATE_INPUT_SIZE = 18
    WORLD = [ShortCorridor, MazeSimulator][1]
    ACTION_SPACE = {ShortCorridor: {0: "R", 1: "L"}, MazeSimulator: {0: "N", 1: "S", 2: "E", 3: "W"}}[WORLD]

    '''
    parameters for the training
    '''
    VISUALIZE_POLICY = True
    NUM_RUNS = 1 # number of times to train a policy from scratch (to handle bad random seed, etc.)
    NUM_EPOCHS = 4000 # an epoch is a complete walkthrough of NUM_EPISODES_PER_EPOCH games
    NUM_EPISODES_PER_EPOCH = 1
    MAX_EPISODE_LEN = 100

    init_actor, init_critic = train_reptile()


    test_env = WORLD()
    test_env.visualize_value(init_critic, -99)
    test_env.visualize(init_actor, -9090)
    print("Goal location of test:\n", test_env)

    for learning_rate in [3e-4]:
        all_runs = []

        for run_iter in range(0, NUM_RUNS):
            print("Starting run number:", run_iter)
            all_runs.append(train_one_run(learning_rate, init_actor, init_critic, test_env))

        # data = np.mean(np.array(all_runs), axis=0)

        # for run in all_runs:
        #     plt.plot(list(range(len(run))), run)
        # plt.savefig("LearningRate%s.png" % (learning_rate))
        # plt.clf()

