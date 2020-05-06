
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

from utils import *

import torch.multiprocessing as mp
import copy
from collections import OrderedDict




class A2C(nn.Module):

    def __init__(self, state_input_size, action_space_size, seed, lr, lr_critic, use_opt=False, ppo=False):
        super(A2C, self).__init__()

        torch.manual_seed(seed)

        self.state_input_size = state_input_size
        self.action_space_size = action_space_size

        # print ("\nstate_input_size: ", self.state_input_size)
        # print ("\naction_space_size: ", self.action_space_size)

        self.ppo = ppo

        self.policy = LSTMActorCriticModel(self.state_input_size, self.action_space_size)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)  #TODO: What is amsgrad

        #state

        #Generate environment
        #Generate_epsiode --> "action_train" for each step
        #Keep track of reward at each step
        #Get the reward from the episode

        #Find policy loss
        #Update parameters (via optimizer)




    def __step(self, env, horizon):
        '''
        makes horizon steps in this trajectory in this environment
        '''
        lam = 0.9

        # number of steps to take in this environment
        if self.ppo:
            env, states, values, actions, rewards, entropie, log_probs = generate_episode(self.old_policy, env, horizon)
        else:
            env, states, values, actions, rewards, entropies, log_probs = generate_episode(self.policy, env, horizon)
        # Rn = self.normalize_advantages(R)

        return env, states, values, actions, rewards, entropies, log_probs

        # traj_len = len(A)

        # # discounted_rewards = [0] * traj_len
        # # reward_tplusone = 0
        # # compute discounted rewards
        # # for t in reversed(list(range(traj_len))):
        # #     discounted_rewards[t] = R[t] + lam * reward_tplusone
        # #     reward_tplusone = discounted_rewards[t]

        # # compute advantage (of that action)
        # td_err = [0]*traj_len
        # adv = [0]*traj_len
        # for t in range(traj_len):
        #     '''
        #     The commented out method may/may not be better, all it does is help propagate reward
        #     farther up the chain
        #     '''
        #     # k = traj_len-t
        #     # if S[t+k] == None:
        #     #     advantages[t] = sum([R[t+i]*lam**(i) for i in range(0, k)]) - self.critic(S[t])
        #     # else:
        #     #     advantages[t] = sum([R[t+i]*lam**(i) for i in range(k)]) + lam**k*self.critic(S[t+k]) - self.critic(S[t])
        #     if S[t+1] == None:
        #         td_err[t] = Rn[t] - self.critic(S[t]).item()
        #         adv[t] = Rn[t] - self.critic(S[t]).item()
        #     else:
        #         td_err[t] = Rn[t] + (lam*self.critic(S[t+1]) - self.critic(S[t])).item()
        #         adv[t] = Rn[t] - self.critic(S[t]).item()
        
        # mini_batch_states = []
        # mini_batch_actions = []
        # mini_batch_td = []
        # mini_batch_adv = []
        # mini_batch_rewards = []
        # for t in range(traj_len):
        #     mini_batch_states.append(S[t].data.numpy())
        #     mini_batch_actions.append(A[t])
        #     mini_batch_td.append(td_err[t])
        #     mini_batch_adv.append(adv[t])
        #     mini_batch_rewards.append(R[t])

        # return env, mini_batch_states, mini_batch_actions, mini_batch_td, mini_batch_adv, mini_batch_rewards, episode

    '''
    For 2D Maze nav task:

    MAML models were trained with up to 500 meta-iterations,
        model with the best avg performance was used for train-time

    policy was trained with MAML to maximize performance after 1 policy gradient update using 20 trajectories
        - thus: a batch size of 20 was used (during meta-train time?/and meta-test?)

    In our evaluation, we compare adaptation to a new task with up to 4 gradient updates, each with 40 samples.
    '''
    def train(self, env, num_batches = 1, batch_size = 2, horizon = 100, batch_envs=None):
        '''
        Train using batch_size samples of complete trajectories, num_batches times (so num_batches gradient updates)
        
        A trajectory is defined as a State, Action, Reward secquence of t steps,
            where t = min(the number of steps to reach the goal, horizon)
        '''

        #Generate environment
        #Generate_epsiode --> "action_train" for each step
        #Keep track of reward at each step
        #Get the reward from the episode

        #Find policy loss
        #Update parameters (via optimizer)


        cumulative_rewards = []
        if self.ppo:
            self.old_policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))

        print ("\nnum_batches: ", num_batches)
        print ("\nbatch_size: ", batch_size)

        for batch in tqdm(range(num_batches)):  #TODO: implement batches

            env = env.generate_fresh()
            env, states, values, actions, rewards, entropies, log_probs = self.__step(env, horizon)

            R = torch.zeros(1,1)
            if rewards[-1] != 0:
                value, _, _ = self.policy((Variable(states[-1]), (self.policy.hx, self.policy.cx)))
                R = value.data

            values.append(Variable(R))
            policy_loss = 0
            value_loss = 0
            gae = torch.zeros(1,1)
            R = Variable(R)

            gamma = 0.9
            tau = 1  #TODO: figure out what this should be

            for i in reversed(range(len(rewards))):
                R = gamma*R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5*advantage.pow(2)

                #Generalized Advantage Estimation
                delta_t = rewards[i] + gamma*values[i+1].data - values[i].data

                gae = gae*gamma*tau + delta_t

                policy_loss = policy_loss - log_probs[i]*Variable(gae)-0.01*entropies[i]

            self.policy.zero_grad()
            loss = (policy_loss + 0.5*value_loss).mean()
            # print ("loss: ", loss)
            loss.backward()
            # ensure_shared_grads(self.policy, shared_model)
            self.optimizer.step()

            cumulative_rewards.append(sum(rewards)/batch_size)

            self.policy.clear_hidden_states()

        world.visualize(model.policy)#, savefile="Heatmap%s" % batch)
        world.visualize_value(model.policy)#, savefile="Valuemap%s" % batch)
    

        return cumulative_rewards

    # #For recurrent policy
    # def update_policy(self):
    #     R = 0
    #     rewards = []

    #     #Discount future rewards back to the present using gamma
    #     for r in reversed(self.policy.reward_episode):
    #         R = r + self.policy.gamma * R
    #         rewards.insert(0, R)

    #     #Scale rewards
    #     rewards = torch.FloatTensor(rewards)
        
    #     #Normalize rewards
    #     rewards = (rewards - rewards.mean()) / (rewards.std()) #+ float(np.info(np.float32).eps))

    #     #Calculate loss
    #     policy_history = torch.stack(self.policy.policy_history)
    #     # print ("HERE: ", policy_history)
    #     loss = (torch.mul(self.policy_history, rewards).mul(-1), -1)  #TODO

    #     #Update network weights
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     #Save and initialize episode history counters
    #     self.policy.loss_history.append(loss.data[0])
    #     self.policy.reward_history.append(np.sum(policy.reward_episode))
    #     self.policy.policy_history.append(self.policy.named_parameters())
    #     self.policy.reset_episode()

maze = [["W", "W", "W", "W", "W", "W", "W", "W", "W"],
        ["W", " ", "G", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
        ["W", "W", "W", "W", "W", "W", "W", "W", "W"]]


maze = [["W", "W", "W", "W"],
        ["W", " ", " ", "W"],
        ["W", " ", "G", "W"],
        ["W", "W", "W", "W"]]

world = MazeSimulator(
                goal_X=2,
                goal_Y=2,
                reward_type="constant",
                state_rep="fullboard",
                maze=maze,
                wall_penalty=0,
                normalize_state=True
            )
model = A2C(world.state_size, world.num_actions, seed=1, lr=0.1, lr_critic=0.1, use_opt=False, ppo=False)

rewards = model.train(world)

print ("rewards: ", rewards)
plt.plot(list(range(len(rewards))), rewards)
plt.savefig("RNNRewards")

world.visualize(model.policy)
world.visualize_value(model.policy)

