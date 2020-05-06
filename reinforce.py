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


        # self.policy = Actor(self.state_input_size, self.action_space_size)
        # self.old_policy = Actor(self.state_input_size, self.action_space_size)

        self.policy = LSTMActorCriticModel(self.state_input_size, self.action_space_size)
        self.old_policy = LSTMActorCriticModel(self.state_input_size, self.action_space_size)


        self.init_optimizers()

    def init_optimizers(self):
        self.opt_a = optim.Adam(self.policy.parameters(), lr=self.lr)

    def compute_loss(self, state, action, weights, ppo_epsilon):
        '''
        weights is what to multiply the log probability by
        '''
        if self.ppo:
            # print ("hx: ", self.policy.hx)
            # print ("cx: ", self.policy.cx)
            _, probs, _ = self.policy((state, (self.policy.hx, self.policy.cx)))
            # print ("output: ", probs)
            # print ("Categorical: ", Categorical(probs))
            _, probs_old, _ = self.old_policy((state, (self.policy.hx, self.policy.cx)))
            logp_ratios =  Categorical(probs).log_prob(action) - Categorical(probs_old).log_prob(action)
            ratios = torch.exp(logp_ratios)
            clipped_adv = torch.clamp(ratios, 1 - ppo_epsilon, 1 + ppo_epsilon) * weights
            non_clipped_adv = ratios * weights
            # theoretically, big = good too keep this from converging too quickly,
            # so we encourage by adding to the thing we are trying to maximize
            # entropy_loss = Categorical(self.policy(state)).entropy()
            return -(torch.min(clipped_adv, non_clipped_adv)).sum(), -Categorical(probs).entropy().sum()
        else:
            ratios =  Categorical(self.policy(state.squeeze(0))).log_prob(action)
            loss = ratios * weights
            return -(loss.sum()), -Categorical(self.policy((state.squeeze(0), (self.hx, self.cx)))).entropy().sum()

    def compute_critic_loss(self, state, value):
        '''
        weights is what to multiply the log probability by
        '''
        value_policy, _, _ = self.policy((state, (self.policy.hx, self.policy.cx)))
        loss = F.smooth_l1_loss(value_policy, value)
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
            env, S, A, R, values, entropies, log_probs = generate_episode(self.old_policy, env, horizon)
        else:
            env, S, A, R, values, entropies, log_probs = generate_episode(self.policy, env, horizon)

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
                # print ("values[t]: ", values[t])
                adv[t] = G - values[t].item()#self.policy.value(S[t]).item()
            critic_target[t] = G
        
        mini_batch_states = []
        mini_batch_actions = []
        mini_batch_td = []
        mini_batch_adv = []
        mini_batch_rewards = []
        # print ("S: ", S)
        for t in range(traj_len):
            mini_batch_states.append(S[t].data.numpy())
            mini_batch_actions.append(A[t])
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
    def train(self, env, batch_envs=None):
        '''
        Train using batch_size samples of complete trajectories, num_batches times (so num_batches gradient updates)
        
        A trajectory is defined as a State, Action, Reward secquence of t steps,
            where t = min(the number of steps to reach the goal, horizon)
        '''
        cumulative_rewards = []
        losses = []
        if self.ppo:
            self.old_policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))

        for batch in tqdm(range(self.args.num_batches)):

            if batch_envs == None:
                parallel_envs = [env.generate_fresh() for _ in range(self.args.batch_size)]
            else:
                assert len(batch_envs) == self.args.batch_size, "supplied envs must match "

            batch_states = []
            batch_actions = []
            batch_td = []
            batch_adv = []
            batch_rewards = []

            q = mp.Queue()
            def multi_process(self, env, horizon, q, done, rank):
                torch.manual_seed(rank)
                env, s, a, td, adv, r = self.__step(env, horizon)
                q.put((s, a, td, adv, r))
                done.wait()

            done = mp.Event()

            num_processes = self.args.batch_size
            self.share_memory()

            processes = []
            for rank in range(num_processes):
                p = mp.Process(target=multi_process, args=(self, parallel_envs[rank], self.args.horizon, q, done, rank))
                p.start()
                processes.append(p)

            for i in range(num_processes):
                (s, a, td, adv, r) = q.get()
                batch_states.extend(s)
                batch_actions.extend(a)
                batch_td.extend(td)
                batch_adv.extend(adv)
                batch_rewards.extend(r)
            
            done.set()
            for p in processes:
                p.join()
            

            # we normalize all of the advantages together, considering over all batches
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
                
                batch_actor_loss, batch_entropy_loss = self.compute_loss(
                                        state=torch.as_tensor(batch_states, dtype=torch.float32)[indices],
                                        action=torch.as_tensor(batch_actions, dtype=torch.float32)[indices],
                                        weights=torch.as_tensor(batch_adv, dtype=torch.float32)[indices],
                                        ppo_epsilon=calc_eps_decay())
                
                if self.use_critic:
                    batch_critic_loss = self.compute_critic_loss(
                                            state=torch.as_tensor(batch_states, dtype=torch.float32)[indices],
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
                    loss.backward(retain_graph=True)

                if self.args.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                
                self.opt_a.step()
            
            if self.ppo:
                # update old policy to the previous new policy
                self.old_policy.load_state_dict(temp_state_dict)

            # if batch % 10 == 0:
            #     print("reward: ", cumulative_rewards[-1])

        # print ("cumulative_rewards: ", cumulative_rewards)
        # print ("losses: ", losses)
        return cumulative_rewards, losses
