"""
Experiment for checking that PPO is working.
"""
import torch.nn as nn
from matplotlib import pyplot as plt
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class MyModel(TorchModelV2, nn.Module):
    def __init__(self, *args, **kwargs):
        TorchModelV2.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        obs_space = args[0]
        action_space = args[1]

        # Action layers
        self.a_layer1 = nn.Linear(obs_space.shape[0], 32)
        self.a_layer2 = nn.Linear(32, 32)
        self.a_layer3 = nn.Linear(32, self.num_outputs)
        self.l_relu = nn.LeakyReLU(0.01)

        # Value layers
        self.v_layer1 = nn.Linear(obs_space.shape[0], 32)
        self.v_layer2 = nn.Linear(32, 32)
        self.v_layer3 = nn.Linear(32, 1)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # Value
        x = self.v_layer1(obs)
        x = self.l_relu(x)
        x = self.v_layer2(x)
        x = self.l_relu(x)
        x = self.v_layer3(x)
        self._value = x

        # Action
        x = self.a_layer1(obs)
        x = self.l_relu(x)
        x = self.a_layer2(x)
        x = self.l_relu(x)
        x = self.a_layer3(x)

        return (x, state)

    def value_function(self):
        return self._value.squeeze(-1)

algo = config.build()

reward_means = []
for _ in range(10):
    results = algo.train()
    reward_mean = results["episode_reward_mean"]
    reward_means.append(reward_mean)

plt.plot(reward_means)
plt.show()
