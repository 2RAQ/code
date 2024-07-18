# Regularized Q-learning through Robust Averaging

The experiments were run in a `python3.10` environment on an `Ubuntu 22.04 LTS` system using an `AMD Ryzen 5950X` 16-Core Processor with 128GB RAM available.

To run the experiments that generated the data used in the paper, set up a `python3.11` environment and install the dependencies provided in the `requirements.txt` file. Then, run the corresponding `collect_x.py` files. A more detailed guide can be found below.

1. `collect_baird.py` for Baird's Experiment
2. `collect_random.py` for the Random Environment
3. `collect_cartpole.py` for the CartPole experiment
4. `collect_lunarlander.py` for the LunarLander experiment

Be aware that all experiments except `collect_lunarlander.py` employ the built-in `multiprocessing` to run all comparison Q-learning methods simultaneously. When running the scripts on a Mac with an M chip, make sure to uncomment the respective lines at the top of each script.

The `plot_x.py` files work the same way and can be used to generate Python plots as well as to save the data as a `.csv` file, which was used to plot the figures in the paper with the `tikz` package, as well as the corresponding metadata as a `.json` file.

1. `plot_baird.py` for Baird's Experiment
2. `plot_random.py` for the Random Environment
3. `plot_cartpole.py` for the CartPole experiment
4. `plot_lunarlander.py` for the LunarLander experiment

All setups have the parameters set as used for the experiments shown in the paper.
