
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal
import logging
logging.basicConfig(filename='goal_locations.log',level=logging.INFO)
import math
import numpy as np

class ActorSmall(nn.Module):

    def __init__(self, state_input_size, action_space_size, hidden_size):
        super(ActorSmall, self).__init__()

        self.input_size = state_input_size
        self.action_space_size = action_space_size

        self.hidden_size = hidden_size
        # self.hidden_size1 = 300
        # self.hidden_size2 = 300

        self.fc1_a = nn.Linear(self.input_size, self.hidden_size)
        self.fc2_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4_a = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc5_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6_a = nn.Linear(self.hidden_size, self.action_space_size)
        self.softmax = nn.Softmax()

        self.fc6_c = nn.Linear(self.hidden_size, 1)

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
        return Categorical(self.softmax(x))

    def value(self, x):
        x = F.relu(self.fc1_a(x))
        x = F.relu(self.fc2_a(x))
        x = F.relu(self.fc3_a(x))
        x = F.relu(self.fc4_a(x))
        # x = F.relu(self.fc5_a(x))
        # x = self.fc6_a(x)
        return self.fc6_c(x)

class ActorContinuous(nn.Module):

    def __init__(self, state_input_size, action_space_size, hidden_size):
        super(ActorContinuous, self).__init__()

        self.input_size = state_input_size
        self.action_space_size = action_space_size

        self.hidden_size = hidden_size

        self.fc1_a = nn.Linear(self.input_size, self.hidden_size)
        self.fc2_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4_a = nn.Linear(self.hidden_size, self.hidden_size)

        self.means = nn.Linear(self.hidden_size, self.action_space_size)
        self.scale = nn.Linear(self.hidden_size, self.action_space_size) # variance

        self.fc6_c = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        '''
        x: input vector describing state
        return: vector containing probabilities?? of each
        '''
        x = F.relu(self.fc1_a(x))
        x = F.relu(self.fc2_a(x))
        x = F.relu(self.fc3_a(x))
        x = F.relu(self.fc4_a(x))

        scale = torch.exp(torch.clamp(self.scale(x), min=math.log(1e-6), max=math.log(10)))
        return Independent(Normal(loc=torch.clamp(self.means(x), -1, 1), scale=scale), 1)

    def value(self, x):
        x = F.relu(self.fc1_a(x))
        x = F.relu(self.fc2_a(x))
        x = F.relu(self.fc3_a(x))
        x = F.relu(self.fc4_a(x))

        return self.fc6_c(x)


def generate_episode(policy, env, T, log=False):
    '''
    return state: list of torch.FloatTensor
           action: list of torch.FloatTensor
           reward: list of floats
    '''
    S, A, R = [], [], []
    for i in range(0, T):
        state = Variable(torch.FloatTensor(env.get_state()))
        m = policy(state)
        action_idx = m.sample()
        # action = policy.ACTION_SPACE[action_idx.item()]
        next_state, reward = env.step(action_idx)
        assert not np.isnan(list(action_idx)[0].item()), "state " + str(state) + " next state " + str(next_state) + " reward " + str(reward)
        # TODO

        S.append(state)
        A.append(action_idx)
        R.append(reward)

        if next_state == None:
            # reached terminal state
            break
        else:
            state = next_state
    
    # if next_state != None:
    #     R[-1] = -100
    if log:
        logging.info(i)
    S.append(torch.FloatTensor(next_state) if next_state != None else None)
    return S, A, R


