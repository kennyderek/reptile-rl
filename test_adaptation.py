

from A2C_PPO import A2C
from sim import MazeSimulator
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm
import torch

def adapt(model, world, num_batches=4, batch_size=20, horizon=100):
    '''
    Adapt a model initialization to a specific world
    @return: rewards
    '''
    return model.train(world, num_batches=1000, batch_size=1, horizon=100)


if __name__ == "__main__":
    saved_model = "reptile_model_init.pth"

    # choose a specific world to adapt the model to


    # world.visualize(model_init.policy)
    # world.visualize_value(model_init.critic)

    init_params = torch.load(saved_model)
    model = A2C(world.state_size, world.num_actions, seed=1, lr=0.1, use_opt=False, ppo=False)

    # TODO
    adapt()