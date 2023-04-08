"""
Experiment to try outputting a distribution over locations to path to instead of direct actions.

CLI Args:
    --eval: Run the last saved policy in the test environment, with visualization.
    --resume: Resume training from the last saved policy.
"""
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import wandb
from gymnasium.spaces.discrete import Discrete
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from torch.distributions import Categorical
from tqdm import tqdm

from mdrc_pacbot_rl.algorithms.rollout_buffer import RolloutBuffer
from mdrc_pacbot_rl.pacman.gym import SemanticPacmanGym as PacmanGym
from mdrc_pacbot_rl.utils import copy_params, get_img_size, init_orthogonal

_: Any

# Hyperparameters
num_envs = 128  # Number of environments to step through at once during sampling.
train_steps = 200  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs
iterations = 1000  # Number of sample/train iterations.
train_iters = 1  # Number of passes over the samples collected.
train_batch_size = 2048  # Minibatch size while training models.
discount = 0.9  # Discount factor applied to rewards.
lambda_ = 0.95  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
eval_steps = 8  # Number of eval runs to average over.
max_eval_steps = 300  # Max number of steps to take during each eval run.
v_lr = 0.001  # Learning rate of the value net.
p_lr = 0.01  # Learning rate of the policy net.
device = torch.device("cpu")

wandb.init(
    project="pacbot",
    entity="mdrc-pacbot",
    config={
        "experiment": "target loc ppo",
        "num_envs": num_envs,
        "train_steps": train_steps,
        "train_iters": train_iters,
        "train_batch_size": train_batch_size,
        "discount": discount,
        "lambda": lambda_,
        "epsilon": epsilon,
        "max_eval_steps": max_eval_steps,
        "v_lr": v_lr,
        "p_lr": p_lr,
    },
    mode="disabled" if len(sys.argv) >= 2 and sys.argv[1] == "--eval" else "offline",
)

v_net_artifact = wandb.Artifact("v_net", "model")
p_net_artifact = wandb.Artifact("p_net", "model")


COMPUTED_DATA_DIR = Path("computed_data")
action_distributions = torch.tensor(
    np.load(COMPUTED_DATA_DIR / "action_distributions.npy"),
    dtype=torch.float32,
)
# embed_dim = 8
# node_embeddings = torch.tensor(
#     np.load(COMPUTED_DATA_DIR / "graph_lap_eigvecs.npy")[:embed_dim].T,
#     dtype=torch.float32,
# )
# node_embeddings /= node_embeddings.std(dim=0, keepdim=True)
node_embeddings = torch.tensor(
    np.load(COMPUTED_DATA_DIR / "node_embeddings.npy"),
    dtype=torch.float32,
)
embed_dim = node_embeddings.shape[1]
with (COMPUTED_DATA_DIR / "node_coords.json").open() as f:
    node_to_coords = [tuple(coords) for coords in json.load(f)]
coords_to_node = {coords: i for i, coords in enumerate(node_to_coords)}
num_nodes = len(node_to_coords)


class ValueNet(nn.Module):
    def __init__(self, obs_shape: torch.Size):
        super().__init__()
        [input_size] = obs_shape
        input_size -= 1  # don't use the pacman_node_index feature
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        # input, _ = make_input_from_obs(input)
        return self.layers(input[:, :-1])


class PolicyNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int):
        super().__init__()
        [input_size] = obs_shape
        input_size -= 1  # don't use the pacman_node_index feature
        self.layers = nn.Sequential(
            # nn.Linear(input_size, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, action_count),
            nn.Linear(input_size, action_count),
        )
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        # return action_distributions[get_nodes_from_obs(input)][:, 4]

        # input, cur_loc_node_indices = make_input_from_obs(input)
        # pellet_probs = self.layers(input)

        # # convert pellet_probs to action probabilities
        # action_probs = torch.matmul(
        #     pellet_probs.unsqueeze(1),
        #     action_distributions[cur_loc_node_indices][:, [4, 13, 276, 285]],
        # ).squeeze(1)

        # input, cur_loc_node_indices = make_input_from_obs(input)
        cur_loc_node_indices = np.array(input[:, -1], dtype=int)
        input = input * 1.0
        input[
            :, :-12
        ] = 0  # debugging: mask everything but valid actions and action to closest ghost
        target_query = self.layers(input[:, :-1])

        action_logits = target_query
        # action_logits += input[:, -7:-2] * 10
        # action_logits -= input[:, -12:-7] * 10
        # action_logits += (input[:, -2] - 0.5) * input[:, -12:-7] * 10
        # action_logits[:, 0] -= 10
        action_probs = nn.functional.softmax(action_logits, dim=-1)
        # with np.printoptions(suppress=True):
        #     print(action_logits)
        #     print(action_probs.numpy(force=True))
        #     print()
        return action_probs


env = SyncVectorEnv([lambda: PacmanGym(random_start=True)] * num_envs)
test_env = PacmanGym(random_start=True)

# If evaluating, just run the eval env
if len(sys.argv) >= 2 and sys.argv[1] == "--eval":
    test_env = PacmanGym(random_start=True, render_mode="human")
    p_net = torch.load("temp/PNet.pt")
    v_net = torch.load("temp/VNet.pt")
    with torch.no_grad():
        obs = torch.Tensor(test_env.reset()[0])
        print(p_net.layers[0].weight[:, -11:])
        print(p_net.layers[0].bias)
        print("actions: [ o  v  ^  <  > ]")
        for _ in range(max_eval_steps):
            action_probs = p_net(obs.unsqueeze(0)).squeeze()
            distr = Categorical(probs=action_probs)
            action = distr.sample().item()
            obs_, reward, done, _, _ = test_env.step(action)
            with np.printoptions(suppress=True, precision=3):
                action_probs = action_probs.numpy()
                print(
                    test_env.game_state.score,
                    f"{reward:.4f}",
                    v_net(obs.unsqueeze(0)).squeeze(),
                    "    \t",
                    action_probs,
                    "\tentropy:",
                    np.nansum(-np.log(action_probs) * action_probs),
                )
            obs = torch.Tensor(obs_)
            if done:
                break
    quit()

# Init PPO
obs_shape = env.envs[0].observation_space.shape
if not obs_shape:
    raise RuntimeError("Observation space doesn't have shape")
obs_size = torch.Size(obs_shape)
act_space = env.envs[0].action_space
if not isinstance(act_space, Discrete):
    raise RuntimeError("Action space was not discrete")
if len(sys.argv) >= 2 and sys.argv[1] == "--resume":
    v_net = torch.load("temp/VNet.pt")
    p_net = torch.load("temp/PNet.pt")
    print("loaded previous model checkpoints")
else:
    v_net = ValueNet(obs_size)
    p_net = PolicyNet(obs_size, int(act_space.n))
    # torch.save(v_net, "temp/VNet.pt")
    # torch.save(p_net, "temp/PNet.pt")
p_net_old = PolicyNet(obs_size, int(act_space.n))
p_net_old.eval()
v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)
buffer = RolloutBuffer(
    obs_size,
    torch.Size((1,)),
    torch.Size((5,)),
    torch.int,
    num_envs,
    train_steps,
    device,
)

obs = torch.Tensor(env.reset()[0])
for _ in tqdm(range(iterations), position=0):
    # Collect experience
    with torch.no_grad():
        for _ in tqdm(range(train_steps), position=1):
            action_probs = p_net(obs)
            actions = Categorical(logits=action_probs).sample().numpy()
            obs_, rewards, dones, truncs, _ = env.step(actions)
            buffer.insert_step(
                obs,
                torch.from_numpy(actions).unsqueeze(-1),
                action_probs,
                rewards,
                dones,
                truncs,
            )
            obs = torch.from_numpy(obs_)
        buffer.insert_final_step(obs)

    # Train
    p_net.train()
    v_net.train()
    copy_params(p_net, p_net_old)

    total_v_loss = 0.0
    total_p_loss = 0.0
    for _ in tqdm(range(train_iters), position=1):
        batches = buffer.samples(train_batch_size, discount, lambda_, v_net)
        for prev_states, actions, action_probs, returns, advantages, _ in batches:
            # print(prev_states.shape, rewards_to_go.shape)
            # print(list(make_input_from_obs(prev_states)[0][:, -1].numpy()))
            # print(list(rewards_to_go.squeeze(axis=-1).numpy()))
            # quit()
            # Train policy network
            with torch.no_grad():
                old_probs = p_net_old(prev_states)
                old_act_probs = Categorical(probs=old_probs).log_prob(actions.squeeze())
            p_opt.zero_grad()
            new_probs = p_net(prev_states)
            new_act_probs = Categorical(probs=new_probs).log_prob(actions.squeeze())
            term1: torch.Tensor = (new_act_probs - old_act_probs).exp() * advantages
            term2: torch.Tensor = (1.0 + epsilon * advantages.sign()) * advantages
            anp = advantages.numpy(force=True)
            p_loss = -term1.min(term2).mean()
            p_loss.backward()
            p_opt.step()
            total_p_loss += p_loss.item()

            # Train value network
            v_opt.zero_grad()
            val_pred = v_net(prev_states)
            val_target = returns  # .unsqueeze(1)
            # diff: torch.Tensor = v_net(prev_states) - rewards_to_go.unsqueeze(1)
            # v_loss = (diff * diff).mean()
            v_loss = nn.functional.mse_loss(val_pred, val_target)

            with np.printoptions(suppress=True):
                print()
                print(anp.mean(), anp.std(), anp.min(), anp.max())
                print(
                    "mean grad magnitude:",
                    p_net.layers[0].weight.grad.abs().numpy(force=True).mean(),
                )
                print(
                    "value net correlation:",
                    np.corrcoef(
                        val_pred.numpy(force=True).squeeze(),
                        val_target.numpy(force=True).squeeze(),
                    )[0, 1],
                )
                #     print()
                #     print(p_net.query.numpy(force=True))
                #     print()
                debug_it = lambda name, arr: (
                    print(f"{name}:".rjust(12), arr.mean(), "+-", arr.std())
                    or print(list(arr))
                )
                # print()
                # print(val_pred.shape, val_target.shape, v_loss.shape)
                # debug_it("val_pred", val_pred.squeeze().numpy(force=True))
                # debug_it("val_target", val_target.squeeze().numpy(force=True))
                # print(v_loss.numpy(force=True))
                # print()
            v_loss.backward()
            v_opt.step()
            total_v_loss += v_loss.item()

    p_net.eval()
    v_net.eval()
    buffer.clear()

    # Evaluate
    eval_done = False
    with torch.no_grad():
        # Visualize
        reward_total = 0
        score_total = 0
        entropy_total = 0.0
        eval_obs = torch.Tensor(test_env.reset()[0])
        for _ in range(eval_steps):
            avg_entropy = 0.0
            steps_taken = 0
            score = 0
            for _ in range(max_eval_steps):
                distr = Categorical(probs=p_net(eval_obs.unsqueeze(0)).squeeze())
                action = distr.sample().item()
                score = test_env.score()
                obs_, reward, eval_done, _, _ = test_env.step(action)
                eval_obs = torch.Tensor(obs_)
                steps_taken += 1
                reward_total += reward
                avg_entropy += distr.entropy()
                if eval_done:
                    eval_obs = torch.Tensor(test_env.reset()[0])
                    break
            avg_entropy /= steps_taken
            entropy_total += avg_entropy
            score_total += score

    wandb.log(
        {
            "avg_eval_episode_reward": reward_total / eval_steps,
            "avg_eval_episode_score": score_total / eval_steps,
            "avg_eval_entropy": entropy_total / eval_steps,
            "avg_v_loss": total_v_loss / train_iters,
            "avg_p_loss": total_p_loss / train_iters,
        }
    )
    print("\navg_eval_episode_score: ", score_total / eval_steps)

    print("saving model checkpoints...")
    torch.save(v_net, "temp/VNet.pt")
    torch.save(p_net, "temp/PNet.pt")
    print("saved")

# Save artifacts
Path("temp/").mkdir(exist_ok=True)

torch.save(v_net, "temp/VNet.pt")
v_net_artifact.add_file("temp/VNet.pt")
wandb.log_artifact(v_net_artifact)

torch.save(p_net, "temp/PNet.pt")
p_net_artifact.add_file("temp/PNet.pt")
wandb.log_artifact(p_net_artifact, "temp/PNet.pt")

wandb.finish()
