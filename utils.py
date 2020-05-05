
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical
import logging
logging.basicConfig(filename='goal_locations.log',level=logging.INFO)


class Recurrent_Actor(nn.Module):

    def __init__(self, state_input_size, action_space_size, hidden_size=128, n_layers=1, dropout_rate=1.0, gamma=0.9):
       super(Recurrent_Actor, self).__init__()
       self.input_size = state_input_size
       self.output_size = action_space_size
       self.hidden_size = hidden_size
       self.n_layers = n_layers

       self.rnn = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)

       self.relu = nn.LeakyReLU()
       self.linear = nn.Linear(self.hidden_size, self.output_size)
       self.softmax = nn.Softmax(dim=-1)

       self.dropout_rate = dropout_rate  #1.0 means no dropout, good values = 0.5-0.8
       self.dropout = nn.Dropout(p=self.dropout_rate)
       self.gamma = gamma  #Typically, 0.8-0.99

       #History
       self.hidden_history = None

       self.policy_history = None
       self.reward_episode = None
       self.reward_episode_local = None

       self.reset_episode()

       #Overall Reward and Loss History
       self.reward_history = list()
       self.reward_history_local = list()
       self.loss_history = list()

    def reset_episode(self):
        #Episode policy and reward history
        self.hidden_history = list()
        self.policy_history = list()
        self.reward_episode = list()
        self.reward_episode_local = list()

    def forward(self, x):
        print ("x: ", x)
        size = x.shape[0]
        print ("size: ", size)
        x = x.view([1, size]) #batch size = 1

        if len(self.hidden_history) > 0:
            h_0 = self.hidden_history[-1]
        else:
            h_0 = None

        x = self.rnn(x, h_0)
        self.hidden_history.append(x)

        x = self.relu(x)
        x = self.linear(x)
        x = self.softmax(x)

        return x   


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

def generate_episode(policy, env, T):
    '''
    return state: list of torch.FloatTensor
           action: list of torch.FloatTensor
           reward: list of floats
    '''
    S, A, R, episode = [], [], [], []
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
        episode.append(env.state_rep_func(env.agent_x, env.agent_y))

        if next_state == None:
            # reached terminal state
            break
        else:
            state = next_state
    
    if next_state != None:
        R[-1] = -100
    logging.info(i)
    S.append(torch.FloatTensor(next_state) if next_state != None else None)
    return S, A, R, episode


