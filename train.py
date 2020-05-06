
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

    f = open("goal_locations.log", "r+")
    f.truncate(0)
    f.close()

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
    # maze = [["W", "W", "W", "W", "W", "W", "W", "W", "W"],
    #         ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", "W", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
    #         ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
    #         ["W", "W", "W", "W", "W", "W", "W", "W", "W"]]
    
    world = MazeSimulator(goal_X=6, goal_Y=10,
                    reward_type="distance",
                    state_rep="fullboard",
                    maze=maze,
                    wall_penalty=-10,
                    normalize_state=True)

    print(world)

    model = REINFORCE(world.state_size, world.num_actions,
                seed=1,
                lr=3e-4,
                use_opt=True,
                ppo=True,
                ppo_epsilon=0.02) # apparently openai suggests 0.2
    rewards, losses = model.train(world, num_batches=200, num_mini_batches=2, batch_size=10, horizon=100)


    world.visualize(model.policy)
    world.visualize_value(model.policy.value)

    plt.plot(list(range(len(losses))), [l[0] for l in losses], c='g', label="Advantages") # policy
    plt.plot(list(range(len(losses))), [l[2] for l in losses], c='b', label="Entropy") # entropy
    plt.plot(list(range(len(losses))), [l[1] for l in losses], c='r', label="Critic") # critic
    leg = plt.legend()
    plt.xlabel("Batch number")
    plt.ylabel("Loss")
    plt.savefig("Losses")
    plt.clf()

    plt.plot(list(range(len(rewards))), rewards)
    plt.xlabel("Batch number")
    plt.ylabel("Reward")
    plt.savefig("Rewards")
    plt.clf()

    f = open("goal_locations.log", "r+")
    goal_found_at = []
    for x in f:
        val = int(x[10:])
        goal_found_at.append(val)
    f.truncate(0)
    plt.plot(list(range(len(goal_found_at))), goal_found_at)
    plt.xlabel("Batch number")
    plt.ylabel("Num timesteps to goal")
    plt.savefig("GoalIndex.png")
    plt.clf()


