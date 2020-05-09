import matplotlib.pyplot as plt
import os
import json
import torch


def update_init_params(target, old, step_size = 0.1):
    """Apply one step of gradient descent on the loss function `loss`, with 
    step-size `step_size`, and returns the updated parameters of the neural 
    network.
    """
    updated = OrderedDict()
    for ((name_old, oldp), (name_target, targetp)) in zip(old.items(), target.items()):
        assert name_old == name_target, "target and old params are different"
        updated[name_old] = oldp + step_size * (targetp - oldp) # grad ascent so its a plus
    return updated


def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def visualize_policy(world, model, folder):
    world.visualize(model.policy, os.path.join(folder, "heatmap"))
    world.visualize_value(model.policy.value, os.path.join(folder, "valuemap"))

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
    if folder != None:
        plt.savefig(os.path.join(folder, "Losses"))
        plt.clf()
    else:
        plt.show()

def plot_rewards(rewards, folder):
    plt.plot(list(range(len(rewards))), rewards)
    plt.xlabel("Batch number")
    plt.ylabel("Reward")
    if folder != None:
        plt.savefig(os.path.join(folder, "Rewards"))
        plt.clf()
    else:
        plt.show()

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

def load_model(model_class, args, path):
    model = model_class(args)
    init_params = torch.load(path)
    model.load_state_dict(init_params)
    return model

def save_data(rewards, losses, folder):
    data = {"rewards": rewards, "losses": losses}
    data_save_path = os.path.join(folder, "data.json")
    with open(data_save_path, 'w') as outfile:
        json.dump(data, outfile)

def compare_parameter_initializations(params_list, model_args, num_steps):
    sample_tasks = [sample_task() for _ in range(num_steps)]
    for d in params_list:
        all_rewards = []
        for t in sample_tasks:
            model = REINFORCE(model_args)
            model.load_state_dict(d["pi"])
            rewards, losses = model.train(t)
            all_rewards.append(rewards)
        d["rewards"] = np.array(all_rewards)

def plot_adaptation(params_list):
    for i in range(len(params_list)):
        d = params_list[i]
        trials = -np.log10(np.abs(d["rewards"]))
    #     trials = d["rewards"]
        x_series = np.array(range(len(trials[0])))
        mean_series = np.mean(trials, axis=0)
        std_err = scipy.stats.sem(trials, axis=0)
        h = std_err * scipy.stats.t.ppf((1.0 + 0.95) / 2.0, len(trials)-1)
        plt.plot(x_series, mean_series, label=d["label"])
        plt.fill_between(x_series, mean_series + h, mean_series - h, alpha=0.2)
    leg = plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Negative log scale reward")
    plt.title("Adaptation speeds of initializations")
    plt.show()