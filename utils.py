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

class ActorWithLSTM(nn.Module):

    def __init__(self, state_input_size, action_space_size):
        super(ActorWithLSTM, self).__init__()

        self.input_size = state_input_size
        self.action_space_size = action_space_size

        self.num_lstm_units = 1 #TODO: check
        self.num_lstm_layers = 1  #TODO: check
        self.batch_size = 1  #TODO: check
        self.hidden_size = 300

        self.history_size = 0

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.num_lstm_units, num_layers=self.num_lstm_layers, batch_first=True)
        self.fc2_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.num_lstm_units, self.action_space_size)
        self.hidden_to_value = nn.Linear(self.num_lstm_units, 1)

        #Init hidden units
        #weights = (num_layers, batch_size, num_lstm_units)
        hidden_a = torch.randn(self.num_lstm_layers, self.batch_size, self.num_lstm_units)
        hidden_b = torch.randn(self.num_lstm_layers, self.batch_size, self.num_lstm_units)

        self.hidden_a = hidden_a
        self.hidden_b = hidden_b

    def init_hidden(self):
        hidden_a = torch.randn(self.num_lstm_layers, self.batch_size, self.num_lstm_units)
        hidden_b = torch.randn(self.num_lstm_layers, self.batch_size, self.num_lstm_units)

        self.hidden_a = hidden_a
        self.hidden_b = hidden_b


    def forward(self, inputs):#x_lengths):
        #Use pack_padded_sequence to make sure the LSTM won't see the padded items

        #TODO: embed, might have reached goal state early

        x, (a, b) = inputs

        if (len(x.size()) > 3):
            x = x.squeeze(0)
        batch_size, seq_len, _ = x.size()

        x_lengths_array = np.array([1 for i in range(batch_size)])
        x_lengths = torch.from_numpy(x_lengths_array)

        #Run through RNN
        #Transform the dimension: (batch_size, seq_len, embedding_dim) --> (batch_size, seq_len, num_lstm_units)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)

        #Run through LSTM
        x, (self.hidden_a, self.hidden_b) = self.lstm(x, (a, b))

        #Undo packing opertaion
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        #Project into output space
        x = x.contiguous()
        x = x.view(-1, x.shape[2]) #reshape the data so it goes into the linear layer

        x = F.relu(self.fc2_a(x))
        x = F.relu(self.fc3_a(x))
        x = F.relu(self.fc4_a(x))

        #run through the actual linear layer
        x = self.hidden_to_output(x)

        #Softmax activation
        #Transform the dimension: (batch_size * seq_len, num_lstm_units) --> (batch_size, seq_len, action_space_size)
        x = F.softmax(x, dim=1) #TODO

        #Reshape back to (batch_size, seq_len, action_space_size)
        x = x.view(batch_size, seq_len, self.action_space_size)

        return x




    def value(self, x):
        if (len(x.size()) > 3):
            x = x.squeeze(0)
        if (len(x.size()) < 3):
            x = x.unsqueeze(0)
            x = x.unsqueeze(0)

        batch_size, seq_len, _ = x.size()

        x_lengths_array = np.array([1 for i in range(batch_size)])
        x_lengths = torch.from_numpy(x_lengths_array)

        #Run through RNN
        #Transform the dimension: (batch_size, seq_len, embedding_dim) --> (batch_size, seq_len, num_lstm_units)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)

        #Run through LSTM
        x, (self.hidden_a, self.hidden_b) = self.lstm(x, (self.hidden_a, self.hidden_b))

        #Undo packing opertaion
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        #Project into output space
        x = x.contiguous()
        x = x.view(-1, x.shape[2]) #reshape the data so it goes into the linear layer

        x = F.relu(self.fc2_a(x))
        x = F.relu(self.fc3_a(x))
        x = F.relu(self.fc4_a(x))
        #run through the actual linear layer - value
        x = self.hidden_to_value(x)#, dim=1)

        return x

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
        self.fc2_a = nn.Linear(self.hidden_size, self.hidden_size1)
        self.fc3_a = nn.Linear(self.hidden_size1, self.hidden_size1)
        self.fc4_a = nn.Linear(self.hidden_size1, self.hidden_size2)
        # self.fc5_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6_a = nn.Linear(self.hidden_size2, self.action_space_size)
        self.softmax = nn.Softmax()

        self.fc6_c = nn.Linear(self.hidden_size2, 1)

        self.fc1_c = nn.Linear(self.input_size, self.hidden_size)



    def forward(self, x):
        '''
        x: input vector describing state
        return: vector containing probabilities?? of each
        '''

        if self.history_size > 0:
            if x.size(0) > (self.history_size+1): #and x.size(0) != self.input_size:
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
        else:
            x = F.relu(self.fc1_a(x))
            x = F.relu(self.fc2_a(x))
            x = F.relu(self.fc3_a(x))
            x = F.relu(self.fc4_a(x))
            # x = F.relu(self.fc5_a(x))
            x = self.fc6_a(x)
            return self.softmax(x)

    def value(self, x):
        #Want value only for the current state (disregarding previous state)

        if self.history_size > 0:
            if x.size(0) == self.input_size:
                x = x
            elif x.size(0) > (self.history_size+1):
                values = x.detach().numpy()
                new_input = []
                for inp in values:
                    new_input.append(inp[0])
                x = torch.from_numpy(np.array(new_input))

            else:
                x = x[0]

            x = F.relu(self.fc1_c(x))
            x = F.relu(self.fc2_a(x))
            x = F.relu(self.fc3_a(x))
            x = F.relu(self.fc4_a(x))
            # x = F.relu(self.fc5_a(x))
            # x = self.fc6_a(x)
            return self.fc6_c(x)

        else:
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

def generate_episode_with_history(policy, env, T, history_length):
    '''
    return state: list of torch.FloatTensor
           action: list of torch.FloatTensor
           reward: list of floats
    '''
    if history_length > 0:
        S, A, R = [], [], []
        for i in range(0, T):
            state = env.state_rep_func(env.agent_x, env.agent_y)#.get_state() #Variable(torch.FloatTensor(env.get_state()))
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

            state_tensor = Variable(torch.from_numpy(np.array(running_state)).float())

            action_probs = policy(state_tensor)
            m = Categorical(action_probs)
            action_idx = m.sample()
            next_state, reward = env.step(action_idx)

            S.append(state_tensor)  
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
    else:
        S, A, R = generate_episode(policy, env, T)
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
