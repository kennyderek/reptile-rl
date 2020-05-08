import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical
import logging
logging.basicConfig(filename='goal_locations.log',level=logging.INFO)


class Actor(nn.Module):

    def __init__(self, state_input_size, action_space_size):
        super(Actor, self).__init__()

        self.input_size = state_input_size
        self.action_space_size = action_space_size

        self.hidden_size = 300
        self.hidden_size1 = 300
        self.hidden_size2 = 300

        self.fc1_a = nn.Linear(self.input_size, self.hidden_size)
        self.fc2_a = nn.Linear(self.hidden_size, self.hidden_size1)
        self.fc3_a = nn.Linear(self.hidden_size1, self.hidden_size1)
        self.fc4_a = nn.Linear(self.hidden_size1, self.hidden_size2)
        # self.fc5_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6_a = nn.Linear(self.hidden_size2, self.action_space_size)
        self.softmax = nn.Softmax()

        self.fc6_c = nn.Linear(self.hidden_size2, 1)


    def forward(self, x):
        '''
        x: input vector describing state
        return: vector containing probabilities?? of each
        '''
        x = F.relu(self.fc1_a(x))
        x = F.relu(self.fc2_a(x))
        x = F.relu(self.fc3_a(x))
        x = F.relu(self.fc4_a(x))
        # x = F.relu(self.fc5_a(x))
        x = self.fc6_a(x)
        return self.softmax(x)

    def value(self, x):
        x = F.relu(self.fc1_a(x))
        x = F.relu(self.fc2_a(x))
        x = F.relu(self.fc3_a(x))
        x = F.relu(self.fc4_a(x))
        # x = F.relu(self.fc5_a(x))
        # x = self.fc6_a(x)
        return self.fc6_c(x)

class ActorWithHistory(nn.Module):

    def __init__(self, state_input_size, action_space_size, history_size=1):
        super(ActorWithHistory, self).__init__()

        self.history_size = history_size
        self.input_size = state_input_size
        self.history_size = self.history_size
        self.action_space_size = action_space_size


        self.hidden_size = 300
        self.hidden_size1 = 300
        self.hidden_size2 = 300

        self.fc1_a = nn.Linear(self.input_size*(self.history_size+1), self.hidden_size)
        self.fc1_c = nn.Linear(self.input_size, self.hidden_size)
        self.fc2_a = nn.Linear(self.hidden_size, self.hidden_size1)
        self.fc3_a = nn.Linear(self.hidden_size1, self.hidden_size1)
        self.fc4_a = nn.Linear(self.hidden_size1, self.hidden_size2)
        # self.fc5_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6_a = nn.Linear(self.hidden_size2, self.action_space_size)
        self.softmax = nn.Softmax()

        self.fc6_c = nn.Linear(self.hidden_size2, 1)


    def forward(self, x):
        '''
        x: input vector describing state
        return: vector containing probabilities?? of each
        '''


        if x.size(0) > (self.history_size+1):
            x = torch.flatten(x, start_dim=1)
        else:
            x = torch.flatten(x)

        x = F.relu(self.fc1_a(x))
        x = F.relu(self.fc2_a(x))
        x = F.relu(self.fc3_a(x))
        x = F.relu(self.fc4_a(x))
        # x = F.relu(self.fc5_a(x))
        x = self.fc6_a(x)
        return self.softmax(x)

    def value(self, x):

        # print ("x: ", x)
        # print ("x.size(0): ", x.size(0))
        # print ("inputs size*history_size: ", self.input_size*(self.history_size+1))
        # print ("input size: ", self.input_size)
        # print ("history size: ", self.history_size)

        if x.size(0) == self.input_size:
            x = x
        elif x.size(0) > (self.history_size+1):
            #x = torch.flatten(x, start_dim=1)
            # print ("YES")
            values = x.detach().numpy()
            new_input = []
            # print ("values: ", values)
            # print (len(values))
            for inp in values:
                new_input.append(inp[0])
            x = torch.from_numpy(np.array(new_input))

        else:
            x = x[0]

        # print ("X AFTER: ", x)

        x = F.relu(self.fc1_c(x))
        x = F.relu(self.fc2_a(x))
        x = F.relu(self.fc3_a(x))
        x = F.relu(self.fc4_a(x))
        # x = F.relu(self.fc5_a(x))
        # x = self.fc6_a(x)
        return self.fc6_c(x)

class Critic(nn.Module):

    def __init__(self, state_input_size, output_space_size):
        super(Critic, self).__init__()

        self.input_size = state_input_size
        self.output_space_size = output_space_size

        self.hidden_size = 300
        self.hidden_size1 = 300
        self.hidden_size2 = 300

        self.fc1_c = nn.Linear(self.input_size, self.hidden_size)
        self.fc2_c = nn.Linear(self.hidden_size, self.hidden_size1)
        self.fc3_c = nn.Linear(self.hidden_size1, self.hidden_size1)
        self.fc4_c = nn.Linear(self.hidden_size1, self.hidden_size2)
        # self.fc5_c = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6_c = nn.Linear(self.hidden_size2, 1)

    def forward(self, x):
        '''
        x is a state, return estimate of its value
        '''

        x = F.relu(self.fc1_c(x))
        x = F.relu(self.fc2_c(x))
        x = F.relu(self.fc3_c(x))
        x = F.relu(self.fc4_c(x))
        # x = F.relu(self.fc5_c(x))
        return self.fc6_c(x)

class ActorCritic(nn.Module):

    def __init__(self, state_input_size, action_space_size):
        super(ActorCritic, self).__init__()

        self.input_size = state_input_size
        self.action_space_size = action_space_size

        self.hidden_size = 100

        # Define the critic
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc5 = nn.Linear(self.hidden_size, self.hidden_size)

        # output layers
        self.softmax_vals = nn.Linear(self.hidden_size, self.action_space_size)
        self.softmax_logits = nn.Softmax()
        
        self.fc6_c = nn.Linear(self.hidden_size, 1)


    def forward(self, x):
        '''
        x is a state, return estimate of its value
        '''

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))

        x = self.softmax_vals(x)
        return self.softmax_logits(x)

    def value(self, x):
        '''
        x is a state, return estimate of its value
        '''

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))

        return self.fc6_c(x)

def generate_episode_with_history(policy, env, T, history_length=1):
    '''
    return state: list of torch.FloatTensor
           action: list of torch.FloatTensor
           reward: list of floats
    '''
    S, A, R = [], [], []
    for i in range(0, T):
        state = env.state_rep_func(env.agent_x, env.agent_y)#.get_state() #Variable(torch.FloatTensor(env.get_state()))
        # print ("\nstate: ", state)
        if i == 0:
            running_state = state
            for j in range(history_length):
                state_pad = np.zeros(len(state))
                running_state = np.vstack((running_state, state_pad))
        else:
            running_state_prev = running_state.copy()
            running_state = state
            for j in range(0, history_length):
                running_state = np.vstack((running_state, running_state_prev[j]))
            # for j in range(history_length-1, 0, -1):
            #     upper_row = running_state[j+1]
            #     running_state[j] = upper_row
            # running_state[0] = state
        # print ("\nrunning_state: ", np.array(running_state))

        # running_state[0] = state
        # running_state[1] = running_state[0].copy()

        state_tensor = Variable(torch.from_numpy(np.array(running_state)).float())
        # print ("\ni: ", i)
        # print ("running_state: ", running_state)
        # print ("state_tensor: ", state_tensor)

        action_probs = policy(state_tensor)
        # print ("action_probs: ", action_probs)
        m = Categorical(action_probs)
        action_idx = m.sample()
        # action = policy.ACTION_SPACE[action_idx.item()]
        # print ("action_idx: ", action_idx)
        next_state, reward = env.step(action_idx)
        # TODO

        S.append(state_tensor)  #TODO: Change this
        A.append(action_idx)
        R.append(reward)

        if next_state == None:
            # reached terminal state
            break
        else:
            state = next_state
    
    if next_state != None:
        R[-1] = -100
    logging.info(i)
    S.append(torch.FloatTensor(next_state) if next_state != None else None)
    return S, A, R

def generate_episode(policy, env, T):
    '''
    return state: list of torch.FloatTensor
           action: list of torch.FloatTensor
           reward: list of floats
    '''
    S, A, R = [], [], []
    for i in range(0, T):
        state = Variable(torch.FloatTensor(env.get_state()))
        action_probs = policy(state)
        m = Categorical(action_probs)
        action_idx = m.sample()
        # action = policy.ACTION_SPACE[action_idx.item()]
        next_state, reward = env.step(action_idx)
        # TODO

        S.append(state)
        A.append(action_idx)
        R.append(reward)

        if next_state == None:
            # reached terminal state
            break
        else:
            state = next_state
    
    if next_state != None:
        R[-1] = -100
    logging.info(i)
    S.append(torch.FloatTensor(next_state) if next_state != None else None)
    return S, A, R
