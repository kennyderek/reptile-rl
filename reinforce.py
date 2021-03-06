
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt
# from tqdm import tqdm
from random import random

from utils import *

import torch.multiprocessing as mp
import copy
from collections import OrderedDict
import math

class REINFORCE(nn.Module):

    def __init__(self, args):
        super(REINFORCE, self).__init__()

        if args.seed != None:
            torch.manual_seed(args.seed)

        # load argument values
        self.args = args
        self.state_input_size = args.state_input_size
        self.action_space_size = args.action_space_size
        self.lr = args.lr
        self.ppo = args.ppo
        self.ppo_base_epsilon = args.ppo_base_epsilon
        self.ppo_dec_epsilon = args.ppo_dec_epsilon
        self.use_critic = args.use_critic
        self.use_entropy = args.use_entropy

        self.policy = args.policy(self.state_input_size, self.action_space_size, args.hidden_size)
        self.old_policy = args.policy(self.state_input_size, self.action_space_size, args.hidden_size)
        self.init_optimizers()

    def init_optimizers(self):
        self.opt_a = optim.Adam(self.policy.parameters(), lr=self.lr)

    def compute_loss(self, state, action, weights, ppo_epsilon):
        '''
        weights is what to multiply the log probability by
        '''
        if self.ppo:
            logp_ratios =  self.policy(state).log_prob(action) - self.old_policy(state).log_prob(action)
            ratios = torch.exp(logp_ratios)
            clipped_adv = torch.clamp(ratios, 1 - ppo_epsilon, 1 + ppo_epsilon) * weights
            non_clipped_adv = ratios * weights

            return -(torch.min(clipped_adv, non_clipped_adv)).sum(), -self.policy(state).entropy().sum()
        else:
            ratios =  self.policy(state).log_prob(action)
            loss = ratios * weights
            return -(loss.sum()), -self.policy(state).entropy().sum()

    def compute_critic_loss(self, state, value):
        '''
        weights is what to multiply the log probability by
        '''
        loss = F.smooth_l1_loss(self.policy.value(state), value)
        return loss.sum()
    
    def normalize_advantages(self, advantages):
        '''
        instead scale advantages between 0 and 1?
        '''
        advantages = np.array(advantages)
        std = np.std(advantages)
        mean = advantages.mean()
        if std != 0:
            advantages = (advantages - mean) / (std)  
        return advantages

    def update_params(self, params, loss, step_size=0.1):
        """
        Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=False)
        for (name, param), grad in zip(params.items(), grads):
            params[name] = param - step_size * grad
        return params

    def __step(self, env, horizon):
        '''
        makes horizon steps in this trajectory in this environment
        '''
        lam = 0.9

        # number of steps to take in this environment
        if self.ppo:
            S, A, R = generate_episode(self.old_policy, env, horizon, self.args.log_goal_locs)
        else:
            S, A, R = generate_episode(self.policy, env, horizon, self.args.log_goal_locs)

        traj_len = len(A)

        # compute advantage (of that action)
        critic_target = [0]*traj_len
        adv = [0]*traj_len
        for t in range(traj_len):
            k = traj_len-t
            G = sum([R[t+i]*lam**(i) for i in range(0, k)])
            if not self.use_critic:
                adv[t] = G
            else:
                adv[t] = G - self.policy.value(S[t]).item()
            critic_target[t] = G
        
        mini_batch_states = []
        mini_batch_actions = []
        mini_batch_td = []
        mini_batch_adv = []
        mini_batch_rewards = []
        for t in range(traj_len):
            mini_batch_states.append(S[t].data.numpy())
            if A[t].dim() == 0:
                mini_batch_actions.append(A[t])
            else:
                assert not np.isnan(list(A[0])[0].item()), "state " + str(S) + " reward " + str(R)
                mini_batch_actions.append(list(A[t]))
            mini_batch_td.append(critic_target[t])
            mini_batch_adv.append(adv[t])
            mini_batch_rewards.append(R[t])

        return env, mini_batch_states, mini_batch_actions, mini_batch_td, mini_batch_adv, mini_batch_rewards

    '''
    For 2D Maze nav task:

    MAML models were trained with up to 500 meta-iterations,
        model with the best avg performance was used for train-time

    policy was trained with MAML to maximize performance after 1 policy gradient update using 20 trajectories
        - thus: a batch size of 20 was used (during meta-train time?/and meta-test?)

    In our evaluation, we compare adaptation to a new task with up to 4 gradient updates, each with 40 samples.
    '''
    def train(self, env, sampler=None):
        '''
        Train using batch_size samples of complete trajectories, num_batches times (so num_batches gradient updates)
        
        A trajectory is defined as a State, Action, Reward secquence of t steps,
            where t = min(the number of steps to reach the goal, horizon)
        '''
        cumulative_rewards = []
        losses = []
        if self.ppo:
            self.old_policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))

        for batch in range(self.args.num_batches):

            if sampler == None:
                parallel_envs = [env.generate_fresh() for _ in range(self.args.batch_size)]
            else:
                parallel_envs = [sampler() for _ in range(self.args.batch_size)]

            batch_states = []
            batch_actions = []
            batch_td = []
            batch_adv = []
            batch_rewards = []

            for rank in range(self.args.batch_size):
                env, s, a, td, adv, r = self.__step(parallel_envs[rank], self.args.horizon)
                batch_states.extend(s)
                batch_actions.extend(a)
                batch_td.extend(td)
                batch_adv.extend(adv)
                batch_rewards.extend(r)

            # we normalize all of the advantages together, considering over all batches
            # pre_norm = copy.deepcopy(batch_adv)
            assert np.sum(np.isnan(np.array(batch_adv))) == 0, str(batch_adv) + "\n" + str(batch_states) +"\n" + str(batch_actions)
            batch_adv = self.normalize_advantages(batch_adv)
            cumulative_rewards.append(sum(batch_rewards)/self.args.batch_size)

            if self.ppo:
                # we make a copy of the current policy to use as the "old" policy in the next iteration
                temp_state_dict = copy.deepcopy(self.policy.state_dict())

            if self.args.random_perm:
                slices = torch.randperm(len(batch_states))
            else:
                slices = torch.range(0, len(batch_states) - 1)

            def calc_eps_decay():
                return self.ppo_base_epsilon + self.args.weight_func(batch) * self.ppo_dec_epsilon

            # lets do minibatches
            slice_len = len(batch_states) // self.args.num_mini_batches
            for m in range(0, self.args.num_mini_batches):
                indices = slices[m*slice_len:(m+1)*slice_len]
                
                state_input = torch.as_tensor(batch_states, dtype=torch.float32)[indices]
                state_input = torch.squeeze(state_input, 1)
                batch_actor_loss, batch_entropy_loss = self.compute_loss(
                                        state=state_input,
                                        action=torch.as_tensor(batch_actions, dtype=torch.float32)[indices],
                                        weights=torch.as_tensor(batch_adv, dtype=torch.float32)[indices],
                                        ppo_epsilon=calc_eps_decay())
                
                if self.use_critic:
                    batch_critic_loss = self.compute_critic_loss(
                                            state=state_input,
                                            value=torch.as_tensor(batch_td, dtype=torch.float32).unsqueeze(1)[indices])
                
                batch_actor_loss = batch_actor_loss
                batch_entropy_loss = (0.1 + self.args.weight_func(batch))*batch_entropy_loss

                loss_d = {"actor": batch_actor_loss.item()}
                loss = batch_actor_loss
                if self.use_critic:
                    loss += batch_critic_loss
                    loss_d["critic"] = batch_critic_loss.item()
                if self.use_entropy:
                    loss += batch_entropy_loss
                    loss_d["entropy"] = batch_entropy_loss.item()
                losses.append(loss_d)

                self.opt_a.zero_grad()
                if m != self.args.num_mini_batches - 1:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()

                if self.args.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                
                self.opt_a.step()
            
            if self.ppo:
                # update old policy to the previous new policy
                self.old_policy.load_state_dict(temp_state_dict)

            if batch % 10 == 0:
                print(cumulative_rewards[-1])

        return cumulative_rewards, losses



