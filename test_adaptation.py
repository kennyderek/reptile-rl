from A2C_PPO import A2C
from sim import MazeSimulator
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm
import torch

# def adapt(model, world, num_batches=4, batch_size=20, horizon=100):
#     '''
#     Adapt a model initialization to a specific world
#     @return: rewards
#     '''
#     return model.train(world, num_batches=1000, batch_size=1, horizon=100)

test_maze = [["W", "W", "W", "W", "W", "W", "W", "W", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", "W", "W", "W", "W", "W", "W", "W", "W"]]

if __name__ == "__main__":
    saved_model = "meta_train_results/best_meta_init_at_iter82.pth"
    # saved_model = "reptile_model_init.pth"
    random.seed(1)

    # choose a specific world to adapt the model to
    world = MazeSimulator(
                goal_X=6,
                goal_Y=1,
                reward_type="distance",
                state_rep="xy",
                maze=test_maze,
                wall_penalty=0,
                normalize_state=True)

    init_params = torch.load(saved_model)
    model = A2C(world.state_size, world.num_actions, seed=1, lr=0.01, use_opt=False, ppo=False)
    model.load_state_dict(init_params)

    rewards = []
    rewards = model.train(world, num_batches=100, batch_size=1, horizon=100)

    print(rewards)

    # world.visualize(model.policy)
    # world.visualize_value(model.critic)


    # adapt()