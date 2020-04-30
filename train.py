
from A2C_PPO import A2C
from reinforce import REINFORCE
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
            ["W", "W", "W", "W", "W", "W", "W", "W", "W"]]
    
    world = MazeSimulator(goal_X=6, goal_Y=1,
                    reward_type="distance",
                    state_rep="xy",
                    maze=maze,
                    wall_penalty=-10,
                    normalize_state=True)

    print(world)

    use_optimizer = True
    if not use_optimizer:
        # this route is faster and cooler b/c we have more control
        model = REINFORCE(world.state_size, world.num_actions, seed=1, lr=0.5, lr_critic=3e-4, use_opt=False, ppo=True)
        rewards, losses = model.train(world, num_batches=200, batch_size=1, horizon=1000)
    else:
        model = REINFORCE(world.state_size, world.num_actions, seed=1, lr=1e-4, lr_critic=1e-5, use_opt=True, ppo=True)
        rewards, losses = model.train(world, num_batches=200, batch_size=5, horizon=100)

    world.visualize(model.policy)
    world.visualize_value(model.policy.value)

    plt.plot(list(range(len(losses))), losses)
    plt.savefig("Losses")
    plt.clf()

    plt.plot(list(range(len(rewards))), rewards)
    plt.savefig("Rewards")

    f = open("goal_locations.log", "r+")
    goal_found_at = []
    for x in f:
        goal_found_at.append(int(x[10:]))
    f.truncate(0)
    plt.plot(list(range(len(goal_found_at))), goal_found_at)
    plt.savefig("GoalIndex.png")


    '''
    These are good settings also!!!
    '''

