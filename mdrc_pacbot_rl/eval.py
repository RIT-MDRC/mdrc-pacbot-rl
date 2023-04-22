"""
Evaluates DQN, vanilla MCTS, and AlphaZero approaches.
"""

from argparse import ArgumentParser
import numpy as np
from pacbot_rs import MCTSContext, PacmanGym
from tqdm import tqdm

import torch
from mdrc_pacbot_rl.pacman.gym import SemanticChannelPacmanGym as PyPacmanGym
from matplotlib import pyplot as plt  # type: ignore


def main():
    parser = ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--time-limit", type=int, default=1000)
    parser.add_argument("--max-nodes", type=int, default=50)
    args = parser.parse_args()
    trials = args.trials
    time_limit = args.time_limit
    max_nodes = args.max_nodes
    net_path = "temp/QNet.ptc"

    # DQN
    dqn_scores = []
    dqn_pellets = []
    with torch.no_grad():
        for _ in tqdm(range(trials)):
            test_env = PyPacmanGym(random_start=False)
            q_net = torch.jit.load(open(net_path, "rb"))
            obs_shape = test_env.observation_space.shape
            if not obs_shape:
                raise RuntimeError("Observation space doesn't have shape")
            obs_, info = test_env.reset()
            action_mask = np.array(list(info["action_mask"]))
            obs = torch.from_numpy(np.array(obs_)).float()
            done = False
            for _ in range(time_limit):
                q_vals = q_net(obs.unsqueeze(0).unsqueeze(0)).squeeze()
                action = (
                    torch.where(torch.from_numpy(action_mask) == 1, -10000, q_vals)
                    .argmax(0)
                    .item()
                )
                obs_, _, done, _, info = test_env.step(action)
                obs = torch.from_numpy(np.array(obs_)).float()
                action_mask = np.array(list(info["action_mask"]))
                if done:
                    break
            dqn_scores.append(test_env.score())
            dqn_pellets.append(test_env.game_state.pellets)

    # MCTS
    mcts = MCTSContext()
    test_env = PacmanGym(False)
    mcts_scores = []
    mcts_pellets = []
    for _ in tqdm(range(trials)):
        test_env.reset()
        done = False
        for _ in range(time_limit):
            action = mcts.ponder_and_choose(test_env, max_nodes, False)
            _, done = test_env.step(action)
            mcts.clear()
            if done:
                break
        mcts_scores.append(test_env.score())
        mcts_pellets.append(test_env.game_state.pellets)

    # AlphaZero
    alpha_scores = []
    alpha_pellets = []
    for _ in tqdm(range(trials)):
        test_env.reset()
        done = False
        for _ in range(time_limit):
            action = mcts.ponder_and_choose(test_env, max_nodes, True)
            _, done = test_env.step(action)
            mcts.clear()
            if done:
                break
        alpha_scores.append(test_env.score())
        alpha_pellets.append(test_env.game_state.pellets)

    # Plot
    fig, axs = plt.subplots(2)

    axs[0].hist(dqn_scores, "auto", alpha=0.3, label="DQN")
    axs[0].hist(mcts_scores, "auto", alpha=0.3, label="MCTS")
    axs[0].hist(alpha_scores, "auto", alpha=0.3, label="AlphaZero")
    axs[0].legend()
    axs[0].set_title("Scores")

    axs[1].hist(dqn_pellets, "auto", alpha=0.3, label="DQN")
    axs[1].hist(mcts_pellets, "auto", alpha=0.3, label="MCTS")
    axs[1].hist(alpha_pellets, "auto", alpha=0.3, label="AlphaZero")
    axs[1].legend()
    axs[1].set_title("Pellets Left")

    plt.show(block=True)


if __name__ == "__main__":
    main()
