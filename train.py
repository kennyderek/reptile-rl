
from A2C_PPO import A2C
from sim import MazeSimulator
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    '''
    Reward type: We use constant reward type if we just want to give -1 at each step,
    otherwise we use negative distance to the goal which is represented by distance

    Wall pentalty: Should be 0 or negative, it penalizes the agent if it hits a wall

    Normalize state: scales the x, y coordinates to be variance of 1 and mean of 0, assuming uniform distribution
    '''

    maze = [["W", "W", "W", "W", "W", "W", "W", "W", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", "W", " ", "W", "W", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
            ["W", "W", "W", "W", "W", "W", "W", "W", "W"]]

    world = MazeSimulator(goal_X=7, goal_Y=10,
                    reward_type="distance",
                    state_rep="xy",
                    maze=maze,
                    wall_penalty=0,
                    normalize_state=True)

    use_optimizer = False
    if not use_optimizer:
        # this route is faster and cooler b/c we have more control
        model = A2C(world.state_size, world.num_actions, seed=1, lr=0.1, use_opt=False, ppo=False)
        rewards = model.train(world, num_batches=1000, batch_size=1, horizon=100)
    else:
        model = A2C(world.state_size, world.num_actions, seed=1, lr=1e-5, use_opt=True, ppo=True)
        rewards = model.train(world, num_batches=1000, batch_size=1, horizon=100)

    world.visualize(model.policy)
    world.visualize_value(model.critic)

    plt.plot(list(range(len(rewards))), rewards)
    plt.savefig("Rewards")



    '''
    These are good settings also!!!
    '''

