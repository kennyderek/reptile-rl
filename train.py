
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
    world = MazeSimulator(goal_X=7, goal_Y=10,
                    reward_type="distance",
                    state_rep="xy",
                    wall_penalty=0,
                    normalize_state=True)
    

    '''
    learning rates for Adam optimizer: 3e-4 is usually good, sometimes it is helpful to step down
    to 1e-4 if it is a larger network

    for non-optimizer gradient ascent: learning rate of 0.1 is actually ok. The depth of the network has a
    much larger impact on the optimal lr now, likely due to the fact of vanishing gradients.
    '''

    '''
    use_opt = True will use Adam, otherwise we just use 
    Is a batch_size of 1 better?! It seems to get to the solution more quickly? It is because it is easier to 
    attribute cause/effect?
    '''
    model = A2C(world.state_size, world.num_actions, seed=1, lr=1e-5, use_opt=True)
    rewards = model.train(world, num_batches=2000, batch_size=5, horizon=100)

    world.visualize(model.policy)
    world.visualize_value(model.critic)

    plt.plot(list(range(len(rewards))), rewards)
    plt.savefig("Rewards")
