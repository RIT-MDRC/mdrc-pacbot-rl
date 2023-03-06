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
## Developer Guidelines
- Add your name to `pyproject.toml`!