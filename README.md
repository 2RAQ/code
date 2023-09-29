# Regularized Q-learning through Robust Averaging

To run the experiments that generated the data, as used in the paper, run the corresponding `collect_x.py` files.

1. `collect_baird.py` for Baird's Experiment
2. `collect_random.py` for the Random Environment
3. `collect_cartpole.py` for the CartPole experiment
4. `collect_lunarlander.py` for the LunarLander experiment

Be aware that all but the `collect_lunar.py` experiments employ the built-in `multiprocessing` to run all comparison Q-learning methods at the same time.

The `plot_x.py` files work the same way and can be used to generate python plots as well as to save the data as a `.csv` file, which was used to plot the figures in the paper with the `tikz` package, as well as the correspodning metadata as `.json` file.

1. `plot_baird.py` for Baird's Experiment
2. `plot_random.py` for the Random Environment
3. `plot_cartpole.py` for the CartPole experiment
4. `plot_lunarlander.py` for the LunarLander experiment


All setups have the parameters set as used for the experiments shown in the paper.

Additional documentation is in the works to enable easier modification of the experimental setups.
