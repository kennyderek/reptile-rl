
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import logging
logging.basicConfig(filename='goal_locations.log',level=logging.INFO)



class LSTMActorCriticModel(nn.Module):
    def __init__(self, num_inputs, action_space_size, hidden_size=300):
        super(LSTMActorCriticModel, self).__init__()
        self.input_size = num_inputs
        self.action_space_size = action_space_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.actor_linear = nn.Linear(self.hidden_size, self.action_space_size)
        self.critic_linear = nn.Linear(self.hidden_size, 1)

        self.actor_softmax = nn.Softmax()

        # self.apply(weights_init(self))  #TODO: impl this
        relu_gain = nn.init.calculate_gain('relu')

        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)

        # self.hx = None
        # self.cx = None

        self.hx = torch.randn((1, 1, self.hidden_size), requires_grad=True)[0]  #TODO: Dimensions
        self.cx = torch.randn((1, 1, self.hidden_size), requires_grad=True)[0]  #TODO: Dimensions


        # self.train()  #TODO: Find this

    def forward(self, inputs):
        inputs, (hx, cx) = inputs

        # print ("inputs: ", inputs)
        x = F.relu(self.fc1(inputs))



        # x, (hx, cx) = inputs
        # print ("x: ", x)
        # print ("x.size(0): ", x.size(0))
        # if x.size(0) == self.input_size:
        #     x = x.view(1, 1, x.size(0))
        #     batch_size = 1
        # x = x.view(-1, x.size(0))
        # else:
        #     x = x.view(x.size(0), 1, self.input_size)
        #     batch_size = x.size(0)

        # print ("x after resize: ", x)


        # hx = hx.view((1, 1, self.hidden_size))
        # cx = cx.view((1, 1, self.hidden_size))

        # print ("hx: ", hx.shape)
        # print ("cx: ", cx.shape)

        # print ("x: ", x.size(0))
        # print ("x: ", x)

        # x = x.view(-1, x.size(0))

        # hx, cx = self.lstm(x, (hx, cx))  #TODO: Can't feed batch to LSTM


        # self.hx = hx  #TODO: Where should I store these?
        # self.cx = cx

        # x = hx

        # actor_x = self.actor_linear(x)
        # critic_x = self.critic_linear(x)

        # return critic_x, F.softmax(actor_x, dim=1), (hx, cx)

        print ("size: ", x.size(0))
        if x.size(0) == self.hidden_size:
            # print ("YES")
            x = x.view(-1, x.size(0))

            hx, cx = self.lstm(x, (hx, cx))  #TODO: Can't feed batch to LSTM


            self.hx = hx  #TODO: Where should I store these?
            self.cx = cx

            x = hx

            actor_x = self.actor_linear(x)
            critic_x = self.critic_linear(x)

            return critic_x, F.softmax(actor_x, dim=1), (hx, cx)
        else:
            print ("NO")
            # x = x[0]
            x = x.squeeze(0)
            for i in range(x.size(0)):
                inp = x[i]
                print ("x size: ", x.size(0))
                print ("x: ", x)
                print ("inp: ", inp.size(0))
                inp = inp.view(-1, inp.size(0))

                hx, cx = self.lstm(inp, (hx, cx))  #TODO: Can't feed batch to LSTM

            # print ("hx_out: ", hx)
            # print ("cx_out: ", cx)

                self.hx = hx  #TODO: Where should I store these?
                self.cx = cx

            x = hx

            actor_x = self.actor_linear(x)
            critic_x = self.critic_linear(x)

            return critic_x, F.softmax(actor_x, dim=1), (hx, cx)

    def clear_hidden_states(self):
        self.hx = torch.randn((self.hidden_size,), requires_grad=True)
        self.cx = torch.randn((self.hidden_size,), requires_grad=True)   


# class Recurrent_Actor(nn.Module):

#     def __init__(self, state_input_size, action_space_size, hidden_size=300, n_layers=1, dropout_rate=1.0, gamma=0.9):
#        super(Recurrent_Actor, self).__init__()
#        self.input_size = state_input_size
#        self.output_size = action_space_size
#        self.hidden_size = hidden_size
#        self.n_layers = n_layers

#        self.rnn = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)

#        self.relu = nn.LeakyReLU()
#        self.linear = nn.Linear(self.hidden_size, self.output_size)
#        self.softmax = nn.Softmax(dim=-1)

#        self.dropout_rate = dropout_rate  #1.0 means no dropout, good values = 0.5-0.8
#        self.dropout = nn.Dropout(p=self.dropout_rate)
#        self.gamma = gamma  #Typically, 0.8-0.99

#        #History
#        self.hidden_history = None

#        self.policy_history = None
#        self.reward_episode = None
#        self.reward_episode_local = None

#        self.reset_episode()

#        #Overall Reward and Loss History
#        self.reward_history = list()
#        self.reward_history_local = list()
#        self.loss_history = list()

#     def reset_episode(self):
#         #Episode policy and reward history
#         self.hidden_history = list()
#         self.policy_history = list()
#         self.reward_episode = list()
#         self.reward_episode_local = list()

#     def forward(self, x):
#         # print ("x: ", x)
#         size = x.shape[0]
#         # print ("size: ", size)
#         if size > 2:
#             x = x.view([1, size*2])

#         else:

#             x = x.view([1, size]) #batch size = 1

#             if len(self.hidden_history) > 0:
#                 h_0 = self.hidden_history[-1]
#             else:
#                 h_0 = None

#             x = self.rnn(x, h_0)
#             self.hidden_history.append(x)

#             x = self.relu(x)
#             x = self.linear(x)
#             x = self.softmax(x)

#         return x   


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
        self.batch_size = batch_size

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
        print ("\nx: ", x)
        print ("\ninput_size: ", self.input_size)  #TODO: batch
        
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
    # S, A, R, episode = [], [], [], []
    states, values, actions, rewards, entropies, log_probs = [], [], [], [], [], []
    state = np.array(env.state_rep_func(env.agent_x, env.agent_y))  #TODO: might need torch.from_numpy.etc
    # print ("INITIAL STATE: ", state)
    # states.append(state)
    running_state = state
    for i in range(0, T+1):
    	running_state = np.vstack((running_state, np.zeros(len(state))))
    for i in range(0, T):
        if i == 0:
            # print ("if state: ", state)
            value, logit, (hx, cx) = policy((Variable(torch.from_numpy(state).float()), (policy.hx, policy.cx)))  #TODO
        else:
            # print ("state: ", state)
            value, logit, (hx, cx) = policy((Variable(state), (policy.hx, policy.cx)))  #TODO
        # prob = F.softmax(logit, dim=1)
        # print ("prob: ", prob)
        log_prob = F.log_softmax(logit, dim=1)
        # print ("log_prob: ", log_prob)
        entropy = -(log_prob * logit).sum(1)
        action = logit.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        state, reward = env.step(action) #TODO: maybe numpy
        # print ("latest state: ", state)
        # running_state = np.vstack((state, running_state))
        # print ("running_state: ", running_state)
        running_state[i] = state
        state = Variable(torch.from_numpy(running_state).float())
        states.append(state)
        values.append(value)
        actions.append(action)
        rewards.append(reward)
        entropies.append(entropy)
        log_probs.append(log_prob)
        # env.hx, env.cx = Variable(hx.data), Variable(cx.data)
        if reward == 100:
            break
    return env, states, actions, rewards, values, entropies, log_probs


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std/torch.sqrt((x**2).sum(1, keepdim=True))
    return x

def weights_init(m):
    weight_shape = list(m.weight.data.size())
    fan_in = weight_shape[1]
    fan_out = weight_shape[0]
    w_bound = np.sqrt(6. / (fan_in + fan_out))
    m.weight.data.uniform_(-w_bound, w_bound)
    m.bias.data.fill_(0)

# def ensure_shared_grads(model):
#     for param in model.parameters():
#         param.__grad = 




#     state = Variable(torch.FloatTensor(env.get_state()))
#     action_probs = policy(state)
#     m = Categorical(action_probs)
#     action_idx = m.sample()
#     # action = policy.ACTION_SPACE[action_idx.item()]
#     next_state, reward = env.step(action_idx)
#     # TODO

#     S.append(state)
#     A.append(action_idx)
#     R.append(reward)
#     episode.append(env.state_rep_func(env.agent_x, env.agent_y))

#     if next_state == None:
#         # reached terminal state
#         break
#     else:
#         state = next_state

# if next_state != None:
#     R[-1] = -100
# logging.info(i)
# S.append(torch.FloatTensor(next_state) if next_state != None else None)
# return S, A, R, episode


