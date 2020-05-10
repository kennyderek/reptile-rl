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
        self.history_size = args.history_size

        # self.policy = ActorWithHistory(self.state_input_size, self.action_space_size, self.history_size)
        # self.old_policy = ActorWithHistory(self.state_input_size, self.action_space_size, self.history_size)

        self.policy = ActorWithLSTM(self.state_input_size, self.action_space_size)
        self.old_policy = ActorWithLSTM(self.state_input_size, self.action_space_size)

        # self.policy = Actor(self.state_input_size, self.action_space_size)
        # self.old_policy = Actor(self.state_input_size, self.action_space_size)
        self.init_optimizers()

    def init_optimizers(self):
        self.opt_a = optim.Adam(self.policy.parameters(), lr=self.lr)

    def compute_loss(self, state, action, weights, ppo_epsilon):
        '''
        weights is what to multiply the log probability by
        '''
        if self.ppo:
            logp_ratios =  Categorical(self.policy(state)).log_prob(action) - Categorical(self.old_policy(state)).log_prob(action)
            ratios = torch.exp(logp_ratios)
            clipped_adv = torch.clamp(ratios, 1 - ppo_epsilon, 1 + ppo_epsilon) * weights
            non_clipped_adv = ratios * weights
            # theoretically, big = good too keep this from converging too quickly,
            # so we encourage by adding to the thing we are trying to maximize
            # entropy_loss = Categorical(self.policy(state)).entropy()
            return -(torch.min(clipped_adv, non_clipped_adv)).sum(), -Categorical(self.policy(state)).entropy().sum()
        else:
            ratios =  Categorical(self.policy(state)).log_prob(action)
            loss = ratios * weights
            return -(loss.sum()), -Categorical(self.policy(state)).entropy().sum()

    def compute_RNN_loss(self, state, hidden_a, hidden_b, action, weights, ppo_epsilon):
        #Flatten all sequences into a REALLY long sequence and calculate the loss on that
        state = Variable(torch.FloatTensor(state))
        # print ("original_state: ", state.size())
        state = state.squeeze(0)
        # print ("state: ", state.size())
        action = Variable(torch.tensor(action, dtype=torch.long))#, dtype=torch.int64))[0][0]
        # print ("RNN LOSS")
        # print ("hidden_a: ", hidden_a)
        hidden_a = Variable(torch.FloatTensor(hidden_a))
        hidden_b = Variable(torch.FloatTensor(hidden_b))
        hidden_a = hidden_a.unsqueeze(0)
        hidden_a = hidden_a.unsqueeze(0)
        hidden_b = hidden_b.unsqueeze(0)
        hidden_b = hidden_b.unsqueeze(0)
        # print ("after hidden_a: ", hidden_a)
        # print ("action: ", action)
        # print ("weights: ", weights)
        if self.ppo:
            logp_ratios =  Categorical(self.policy((state, (hidden_a, hidden_b)))).log_prob(action) - Categorical(self.old_policy((state, (hidden_a, hidden_b)))).log_prob(action)
            # print ("logp_ratios: ", logp_ratios)
            ratios = torch.exp(logp_ratios)#.to(dtype=torch.long)
            # print ("ratios: ", ratios)
            # print ("ppo_epsilon: ", ppo_epsilon)
            clipped_adv = ratios[0][0].item()#torch.tensor((torch.clamp(ratios, 1 - ppo_epsilon, 1 + ppo_epsilon) * weights))
            if clipped_adv < (1-ppo_epsilon):
                clipped_adv = torch.tensor((1-ppo_epsilon)*weights)
            elif clipped_adv > (1+ppo_epsilon):
                clipped_adv = torch.tensor((1+ppo_epsilon)*weights)
            else:
                clipped_adv = torch.tensor(clipped_adv)
            non_clipped_adv = torch.tensor((ratios[0][0].item() * weights[0]))
            # print ("clipped_adv: ", clipped_adv)
            # print ("non_clipped_adv: ", non_clipped_adv)
            # theoretically, big = good too keep this from converging too quickly,
            # so we encourage by adding to the thing we are trying to maximize
            entropy_loss = Categorical(self.policy((state, (hidden_a, hidden_b)))).entropy().sum()
            # print ("entropy loss: ", entropy_loss)
            return -(torch.min(clipped_adv, non_clipped_adv)), -Categorical(self.policy((state, (hidden_a, hidden_b)))).entropy().sum()
        else:
            ratios =  Categorical(self.policy((state, (hidden_a, hidden_b)))).log_prob(action)
            loss = ratios * weights
            return -(loss.sum()), -Categorical(self.policy((state, (hidden_a, hidden_b)))).entropy().sum()

    def train(self, env, batch_envs=None):
        '''
        Train using batch_size samples of complete trajectories, num_batches times (so num_batches gradient updates)
        
        A trajectory is defined as a State, Action, Reward secquence of t steps,
            where t = min(the number of steps to reach the goal, horizon)
        '''
        # print ("starting env: ", env)
        cumulative_rewards = []
        losses = []
        env = env.generate_fresh()
        if self.ppo:
            self.old_policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))

        total_batch_states = []
        total_batch_actions = []
        total_batch_td = []
        total_batch_advs = []
        total_batch_rewards = []
        total_hidden_a = []
        total_hidden_b = []
        print ("num_batches: ", self.args.num_batches)
        for batch in tqdm(range(self.args.num_batches)):  #batch_size will always be 1, horizon will also always be 1
            # print ("batch: ", batch)
            #TODO: This is a question
            #"Backpropagating return-weighted eligibilities affects the policy such that it makes histories that were better than 
            #other histories (in terms of reward) more likely by reinforcing the probabilities of taking similar actions for similar histories."

            # print ("STEPING...")
            env, s, a, td, adv, r, hidden_a, hidden_b = self.__step_RNN(env, self.args.horizon)
            # print ("DONE STEPPING")
            # print ("env after step: ", env)

            total_batch_states.append(s)
            total_batch_actions.append(a)
            total_batch_td.append(td)
            total_batch_advs.append(adv)
            total_batch_rewards.append(r)
            total_hidden_a.append(hidden_a)
            total_hidden_b.append(hidden_b)

            batch_state = s
            batch_action = a
            batch_td = td
            batch_adv = adv
            batch_reward = r

            # print ("batch_adv: ", batch_adv)
            # print ("batch_reward: ", batch_reward)

            # print ("batch_state: ", batch_state)
            #TODO = normalize batch advantage
            def calc_eps_decay():
                return self.ppo_base_epsilon + self.args.weight_func(batch) * self.ppo_dec_epsilon


            #End of "episode": add reward, calculate loss, clear hidden state
            if (batch % 500) == 0:
                print ("\nbatch: ", batch)
                # print ("total_batch_rewards: ", total_batch_rewards)
                if self.ppo:
                    temp_state_dict = copy.deepcopy(self.policy.state_dict())
                cumulative_rewards.append(np.mean(total_batch_rewards))
                
                #Compute advantage of that action
                lam = 0.9
                episode_length = len(total_batch_actions)
                advantages = [0]*episode_length
                tds = [0]*episode_length
                for t in range(episode_length):
                    k = episode_length-t
                    # print ("total_batch_rewards: ", total_batch_rewards[t])
                    G = sum(total_batch_rewards[t+i][0]*lam**(i) for i in range(0, k))
                    if not self.use_critic:
                        advantages[t] = G
                    else:
                        t_state = Variable(torch.FloatTensor(total_batch_states[t]))
                        t_state = t_state.squeeze(0)
                        advantages[t] = G - self.policy.value(t_state)
                    tds[t] = G

                for i in range(len(total_batch_states)):
                    # print ("total_hidden_a[i]: ", total_hidden_a[i])
                    batch_actor_loss, batch_entropy_loss = self.compute_RNN_loss(total_batch_states[i], total_hidden_a[i], total_hidden_b[i], total_batch_actions[i], advantages[i], ppo_epsilon=calc_eps_decay())
                    # print ("HERE")
                    if self.use_critic:
                        batch_critic_loss = self.compute_critic_loss(total_batch_states[i], tds[i])

                    # print ("ONE")
                    batch_actor_loss = batch_actor_loss
                    batch_entropy_loss = (0.1 + self.args.weight_func(batch))*batch_entropy_loss

                    loss_d = {"actor": batch_actor_loss.item()}
                    loss = batch_actor_loss
                    # print ("loss: ", loss)
                    if self.use_critic:
                        loss += batch_critic_loss
                        loss_d["critic"] = batch_critic_loss.item()
                    if self.use_entropy:
                        loss += batch_entropy_loss
                        loss_d["entropy"] = batch_entropy_loss.item()
                    losses.append(loss_d)

                    # print ("TWO")

                    self.opt_a.zero_grad()
                    loss.backward(retain_graph=True)

                    if self.args.gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    
                    self.opt_a.step()
            
                if self.ppo:
                    # update old policy to the previous new policy
                    self.old_policy.load_state_dict(temp_state_dict)

                # self.policy.init_hidden()
                # if self.ppo:
                #     self.old_policy.init_hidden()

                # print ("THREE")

                env = env.generate_fresh()

                total_batch_states = []
                total_batch_actions = []
                total_batch_td = []
                total_batch_advs = []
                total_batch_rewards = []

                # print ("FOUR")

                # print(cumulative_rewards[-1])

        return cumulative_rewards, losses


    def compute_critic_loss(self, state, value):
        '''
        weights is what to multiply the log probability by
        '''
        # print ("self.policy.value(state): ", self.policy.value(state))
        # print ("value: ", value)
        state = Variable(torch.FloatTensor(state))
        state = state.squeeze(0)
        # state = state.squeeze(0)
        # print ("state: ", state)
        # print ("policy value: ", self.policy.value(state))
        value = Variable(torch.FloatTensor([value]))
        # print ('value: ', value)
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

    def __step_RNN(self, env, horizon):
        '''
        makes horizon steps in this trajectory in this environment
        '''
        # print ("in step: ", env)
        lam = 0.9

        # number of steps to take in this environment
        if self.ppo:
            S, A, R, Hidden_A, Hidden_B = generate_episode_LSTM(self.old_policy, env, horizon, self.old_policy.history_size)#generate_episode_with_history(self.old_policy, env, horizon, self.old_policy.history_size)
            # print ("All Rewards: ", R)
        else:
            S, A, R, Hidden_A, Hidden_B = generate_episode_LSTM(self.old_policy, env, horizon, self.old_policy.history_size)#generate_episode_with_history(self.policy, env, horizon, self.policy.history_size)

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
            mini_batch_actions.append(A[t])
            mini_batch_td.append(critic_target[t])
            mini_batch_adv.append(adv[t])
            mini_batch_rewards.append(R[t])

        return env, mini_batch_states, mini_batch_actions, mini_batch_td, mini_batch_rewards, mini_batch_rewards, Hidden_A, Hidden_B #mini_batch_adv, mini_batch_rewards


    def __step(self, env, horizon):
        '''
        makes horizon steps in this trajectory in this environment
        '''
        # print ("in step: ", env)
        lam = 0.9

        # number of steps to take in this environment
        if self.ppo:
            S, A, R = generate_episode_LSTM(self.old_policy, env, horizon, self.old_policy.history_size)#generate_episode_with_history(self.old_policy, env, horizon, self.old_policy.history_size)
            # print ("All Rewards: ", R)
        else:
            S, A, R = generate_episode_LSTM(self.old_policy, env, horizon, self.old_policy.history_size)#generate_episode_with_history(self.policy, env, horizon, self.policy.history_size)

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
            mini_batch_actions.append(A[t])
            mini_batch_td.append(critic_target[t])
            mini_batch_adv.append(adv[t])
            mini_batch_rewards.append(R[t])

        return env, mini_batch_states, mini_batch_actions, mini_batch_td, mini_batch_rewards, mini_batch_rewards #mini_batch_adv, mini_batch_rewards

    '''
    For 2D Maze nav task:
    MAML models were trained with up to 500 meta-iterations,
        model with the best avg performance was used for train-time
    policy was trained with MAML to maximize performance after 1 policy gradient update using 20 trajectories
        - thus: a batch size of 20 was used (during meta-train time?/and meta-test?)
    In our evaluation, we compare adaptation to a new task with up to 4 gradient updates, each with 40 samples.
    '''
    def train_original(self, env, batch_envs=None):
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