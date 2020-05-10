# from A2C_PPO import A2C
from reinforce import REINFORCE
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

test_vertical_maze = [["W", "W", "W", "W", "W", "W", "W", "W", "W"],
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

vertical_goal = (6, 1)

test_horizontal_maze = [["W", "W", "W", "W", "W", "W", "W", "W", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", "W", "W", "W", "W", "W", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
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

horizontal_goal = (1, 6)

test_random_maze = [['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'], 
            ['W', ' ', 'W', ' ', ' ', ' ', ' ', ' ', ' ', 'W', 'W'], 
            ['W', ' ', ' ', ' ', ' ', 'W', ' ', ' ', ' ', ' ', 'W'], 
            ['W', ' ', ' ', ' ', 'W', ' ', ' ', ' ', ' ', 'G', 'W'], 
            ['W', 'W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'], 
            ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'], 
            ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'], 
            ['W', ' ', 'W', ' ', 'W', ' ', 'W', ' ', ' ', ' ', 'W'], 
            ['W', ' ', ' ', 'W', ' ', ' ', ' ', 'W', ' ', ' ', 'W'], 
            ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W'], 
            ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']]

random_goal = (9, 3)

if __name__ == "__main__":
    # saved_model = "meta_train_results/best_meta_init_at_iter82.pth"
    random.seed(1)

    # choose a specific world to adapt the model to
    world = MazeSimulator(
                goal_X=random_goal[0],
                goal_Y=random_goal[1],
                reward_type="distance",
                state_rep="fullboard",
                maze=test_random_maze,
                wall_penalty=0,
                normalize_state=True)
    class Args():   
        def __init__(self, world):
            # type of model related arguments
            self.seed = 1
            self.state_input_size = world.state_size
            self.action_space_size = world.num_actions
            self.lr = 3e-4
            self.ppo = True
            self.ppo_base_epsilon = 0.1
            self.ppo_dec_epsilon = 0.1
            self.use_critic = True
            self.use_entropy = True

            # training related arguments
            self.gradient_clipping = True
            self.random_perm = True
            self.num_batches = 300
            self.num_mini_batches = 2
            self.batch_size = 10
            self.horizon = 100
            self.weight_func = lambda batch_num: (1 - batch_num/self.num_batches)**2
            self.history_size = 0

    args = Args(world)
    args.ppo = True
    args.use_critic = True
    args.use_entropy = True
    args.gradient_clipping = False
    args.ppo_base_epsilon = 0.2
    args.ppo_dec_epsilon = 0

    # model = REINFORCE(args)
    saved_model = "meta_train_results/final_random_0_reptile_model_init.pth"
    init_params = torch.load(saved_model)
    # model = A2C(world.state_size, world.num_actions, seed=1, lr=0.01, use_opt=False, ppo=False)
    model = REINFORCE(args)
    model.load_state_dict(init_params)

    rewards = []
    rewards, losses = model.train(world)#, num_batches=100, batch_size=1, horizon=100)

    print(rewards)


    plt.plot(list(range(len(rewards))), rewards)
    plt.savefig("TestRewardsOfReptile_Random_0")



    # world.visualize(model.policy)
    world.visualize_value(model.policy, "TestValuemap_Random_1")


    # adapt()