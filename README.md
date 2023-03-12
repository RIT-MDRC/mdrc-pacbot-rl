# RL Experiments for Pacbot
This repository contains experiments for the high level planning portion of our
Pacbot robot. Since RL is a finicky beast and there's a hundred things we
_could_ do in the small amount of time we have, our strategy should be to start
from a baseline and make incremental, quantifiable improvements.
## Repo Structure
- `mdrc_pacbot_rl`: Source code.
    - `algorithms`: Common algorithms used across experiments.
    - `experiments`: Single file experiments.
    - `pacman`: Environment(s) for playing Pacman.
    - `utils.py`: Miscellaneous utilities.
- `tests`: Automated tests.
## Quickstart
1. Install [Poetry](https://python-poetry.org/) and make sure you have Python 3.9 or later.
2. Run `poetry install`.
3. Use `poetry shell` to enter the new environment you just created.
4. Choose an experiment to run. For example, `random_actions` runs the Pacman gym environment with visuals.
5. Run the experiment with `python mdrc_pacbot_rl/experiments/{NAME_OF_EXPERIMENT}.py`.
## WandB Logging
Our repo supports using [Weights and Biases](https://wandb.ai) to enable
realtime logging and collaboration. We can also easily compare the results of
experiments, and track well performing models.

First, make sure you're invited to our organization. Make an account and ask in
the Slack/Discord channel to be added.

Once you're in, you should see that we have two projects: `pacbot` and `tests`.
`pacbot` is stuff we've logged for our main RL experiments, while `tests` just
contains run data for stuff not directly related. For example, there's plots
that show how our PPO implementation performs against Cartpole. Click on `tests`.

You should see multiple runs and their plots. Click the grouping button and
group by "experiment". This should get you a direct comparison of how each
experiment performs on average.

You'll also see the option to group by group. We use groups when performing
hyperparameter searches, like with population based training.
## Developer Guidelines
- Add your name to `pyproject.toml`!