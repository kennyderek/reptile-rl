
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):

    def __init__(self, state_input_size, action_space_size):
        super(Actor, self).__init__()

        self.input_size = state_input_size
        self.action_space_size = action_space_size

        self.hidden_size = 300

        self.fc1_a = nn.Linear(self.input_size, self.hidden_size)
        self.fc2_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4_a = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc5_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6_a = nn.Linear(self.hidden_size, self.action_space_size)
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
        # x = F.relu(self.fc5_a(x))
        x = self.fc6_a(x)
        return self.softmax(x)

class Critic(nn.Module):

    def __init__(self, state_input_size, output_space_size):
        super(Critic, self).__init__()

        self.input_size = state_input_size
        self.output_space_size = output_space_size

        self.hidden_size = 300

        # Define the critic
        self.fc1_c = nn.Linear(self.input_size, self.hidden_size)
        self.fc2_c = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_c = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4_c = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc5_c = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6_c = nn.Linear(self.hidden_size, output_space_size)

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
    
    S.append(torch.FloatTensor(next_state) if next_state != None else None)
    return S, A, R


