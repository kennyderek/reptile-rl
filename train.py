
from reinforce import REINFORCE
from sim import MazeSimulator
import matplotlib.pyplot as plt
import os
import json
import torch

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def visualize_policy(model, folder):
    # world.visualize(model.policy, os.path.join(folder, "heatmap"))
    world.visualize_value(model.policy, os.path.join(folder, "valuemap"))

def plot_losses(losses, folder):
    if "actor" in losses[0]:
        plt.plot(list(range(len(losses))), [l["actor"] for l in losses], c='g', label="Actor") # policy
    if "entropy" in losses[0]:
        plt.plot(list(range(len(losses))), [l["entropy"] for l in losses], c='b', label="Entropy") # entropy
    if "critic" in losses[0]:
        plt.plot(list(range(len(losses))), [l["critic"] for l in losses], c='r', label="Critic") # critic
    leg = plt.legend()
    plt.xlabel("Batch number")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(folder, "Losses"))
    plt.clf()

def plot_rewards(rewards, folder):
    plt.plot(list(range(len(rewards))), rewards)
    plt.xlabel("Batch number")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(folder, "Rewards"))
    plt.clf()

def plot_goal_loc(folder):
    f = open("goal_locations.log", "r+")
    goal_found_at = []
    for x in f:
        val = int(x[10:])
        goal_found_at.append(val)
    f.truncate(0)
    plt.plot(list(range(len(goal_found_at))), goal_found_at)
    plt.xlabel("Batch number")
    plt.ylabel("Num timesteps to goal")
    plt.savefig(os.path.join(folder, "GoalIndex.png"))
    plt.clf()

def save_model(model, folder, model_name):
    torch.save(model.state_dict(), os.path.join(folder, model_name))


maze = [["W", "W", "W", "W", "W", "W", "W", "W", "W"],
        ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
        ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
        ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
        ["W", " ", "W", " ", " ", " ", " ", " ", "W"],
        ["W", " ", "W", " ", " ", " ", " ", " ", "W"],
        ["W", " ", "W", "W", "W", "W", "W", " ", "W"],
        ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
        ["W", " ", " ", " ", "W", " ", " ", "W", "W"],
        ["W", " ", "W", "W", "W", " ", "W", " ", "W"],
        ["W", " ", "W", " ", " ", " ", " ", " ", "W"],
        ["W", " ", "W", " ", " ", " ", " ", " ", "W"],
        ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
        ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
        ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
        ["W", "W", "W", "W", "W", "W", "W", "W", "W"]]

maze_goal = (6, 10)

vertical_maze = [["W", "W", "W", "W", "W", "W", "W", "W", "W"],
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

horizontal_maze = [["W", "W", "W", "W", "W", "W", "W", "W", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", "W", "W", "W", "W", " ", " ", " ", "W"],
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

simple_maze = [["W", "W", "W", "W"],
        ["W", " ", " ", "W"],
        ["W", " ", "G", "W"],
        ["W", "W", "W", "W"]]


simple_goal = (2, 2)

horizontal_goal = (1, 6)

if __name__ == "__main__":
    
    '''
    Reward type: We use constant reward type if we just want to give -1 at each step,
    otherwise we use negative distance to the goal which is represented by distance
    Wall pentalty: Should be 0 or negative, it penalizes the agent if it hits a wall
    Normalize state: scales the x, y coordinates to be variance of 1 and mean of 0, assuming uniform distribution
    '''
    
    world = MazeSimulator(goal_X=simple_goal[0], goal_Y=simple_goal[1],
                    reward_type="distance",
                    state_rep="fullboard",
                    maze=simple_maze,
                    wall_penalty=-10,
                    normalize_state=True)  #6, 10

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

            #For training LSTM
            self.num_batches = 30000
            self.num_mini_batches = 1#2
            self.batch_size = 1#10
            self.horizon = 1#100

            #For training other policies
            # self.num_batches = 300
            # self.num_mini_batches = 2
            # self.batch_size = 10
            # self.horizon = 100

            self.weight_func = lambda batch_num: (1 - batch_num/self.num_batches)**2
            self.history_size = 1
    
    def run_experiment(args, folder):
        model = REINFORCE(args)
        # rewards, losses = model.train(world)
        rewards, losses = model.train_RNN(world)
        make_folder(folder)
        plot_losses(losses, folder)
        plot_rewards(rewards, folder)
        plot_goal_loc(folder)
        visualize_policy(model, folder)
        data = {"rewards": rewards, "losses": losses}
        data_save_path = os.path.join(folder, "data.json")
        with open(data_save_path, 'w') as outfile:
            json.dump(data, outfile)
        save_model(model, folder, "model.pth")

    # clear goal locations log at start
    f = open("goal_locations.log", "r+")
    f.truncate(0)
    f.close()

    '''
    Make comparison of more high-level ideas
    '''
    # set args for Vanilla REINFORCE & run
    # a = Args(world)
    # a.ppo = False
    # a.use_critic = False
    # a.use_entropy = False
    # a.gradient_clipping = False
    # run_experiment(a, "baseline_REINFORCE")
    
    # set args for REINFORCE with critic & run
    # a = Args(world)
    # a.ppo = False
    # a.use_critic = True
    # a.use_entropy = False
    # a.gradient_clipping = False
    # run_experiment(a, "critic_baseline_REINFORCE")

    # set args for REINFORCE with entropy and critic & run
    # a = Args(world)
    # a.ppo = False
    # a.use_critic = True
    # a.use_entropy = True
    # a.gradient_clipping = False
    # run_experiment(a, "critic_entropy_REINFORCE")

    # set args for REINFORCE with ppo, entropy and critic & run
    a = Args(world)
    a.ppo = True
    a.use_critic = True
    a.use_entropy = True
    a.gradient_clipping = False
    a.ppo_base_epsilon = 0.2
    a.ppo_dec_epsilon = 0
    run_experiment(a, "new_critic_entropy_ppo_REINFORCE")


    '''
    More specialized random techniques?
    '''
    # setting gradient clipping to true
    # setting decreasing ppo epsilon function
    # setting random perumations to True or False
    # 
