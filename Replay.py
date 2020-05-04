#Adapted from https://gist.github.com/awjuliani/35d2ab3409fc818011b6519f0f1629df and
#https://github.com/Bigpig4396/PyTorch-Deep-Recurrent-Q-Learning-DRQN/blob/master/DRQN.py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable

class ReplayMemory(object):
    def __init__(self, memory_size = 1000, batch_size=1):
        self.memory = []
        self.memory_size = memory_size

        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.cell = torch.nn.LSTMCell(self.input_size, self.hidden_size)  #TODO: Is this correct?
        # self.cellT = torch.nn.LSTMCell(self.input_size, self.hidden_size)
    
    def add(self, episodes): #from REPTILE_training train
        num_episodes = len(episodes.keys())
        if num_episodes > abs(self.memory_size - len(self.memory)):
            difference = num_episodes - (abs(self.memory_size - len(self.memory)))
            self.memory = self.memory[difference: len(self.memory)]
        for key in episodes.keys():
            episode = [episodes[key]]
            num_episodes.append(episode)

        # if len(self.memory) + 1 >= self.memory_size:
        #     self.memory[0:(1+len(self.memory))-self.memory_size] = []
        # self.memory.append(experience)


    '''
    Bootstrapped Random Updates: 
        Agent doesn't necessarily have to be at the start of the maze, can be anywhere during the selected episode.
    '''        
    def sample(self, trace_length=100):
        if self.is_available():
            sample_index = random.randint(0, len(self.memory)-2)  #Prevent using most recent episodes
            episode = self.memory[sample_index]
            episode_length = len(self.memory[sample_index])
            start_point = random.randint(0, episode_length-2) #Use at least 2 frames
            return episode[start_point:episode_length]
        return []


    def is_available(self):
        return len(self.memory) > 0

    def __str__(self):
        for i in range(len(self.memory)):
            print ('Episode', i, 'length', len(self.memory[i]))

    '''
    TODO: Bootstrapped Sequential
    '''


class RNN(nn.Module):
    def __init__(self, num_actions, state_size):
        super(RNN, self).__init__()
        self.input_size = 16 #for LSTM
        self.hidden_size = 16 #for LSTM
        self.num_layers = 1
        self.num_actions = num_actions #Also output size and input size
        self.first_input_size = state_size
        self.fc1 = nn.Linear(self.first_input_size, self.hidden_size)
        self.flat1 = Flatten()
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.num_actions)

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-13)


    def forward(self, x, hidden):
        h1 = F.relu(self.fc1(x))
        h2 = self.flat1(h1)
        h3 = h2.unsqueeze(1)
        h3, new_hidden = self.lstm(h2, hidden)
        h4 = F.relu(self.fc1(hc3))
        h5 = self.fc2(h4)
        return h5, new_hidden

    #meant for one-hot encoding 
    def array_to_tensor(self, array):
        # numpy_array = np.array(array)
        return torch.from_array(array)

    def array_list_to_batch(self, x):
        temp_batch = self.array_to_tensor(x[0])
        temp_batch = temp_batch.unsqueeze(0)
        for i in range(1, len(x)):
            tensor = self.array_to_tensor(x[i])
            tensor = tensor.unsqueeze(0)
            temp_batch = torch.cat([temp_batch, tensor], dim=0)
        return temp_batch

    #can incorporate greedy action via epsilon
    def get_action(self, obs, hidden, epsilon):
        policy, new_hidden = self.forward(self.array_to_tensor(obs).unsqueeze(0), hidden)
        action = policy[0].max(1)[1].data[0].item()
        



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
