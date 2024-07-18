from __future__ import annotations

import csv
import json
from pathlib import Path

from absl import logging
import matplotlib.pyplot as plt
import numpy as np

from agents.data_manager import DataManager

logging.set_verbosity(logging.DEBUG)

dm = DataManager()
dm.load_data("data/random_environment_vr_full")


def save_random_as_csv(
    dm: DataManager,
    experiments: list[str],
    n_environments: int,
    path: str = "data",
    data_range: None | tuple[int, int] = None,
    samplerate: int = 1000,
) -> None:
    """Saves random environment data to CSV files with accompanying metadata in
    a JSON file.

    Extracts mean and standard deviation values from the specified experiments in
    the data manager for each environment, and saves these values in separate CSV
    files for each environment. The metadata for each experiment is saved in a
    single JSON file.

    Args:
        dm: The data manager instance containing the raw data.
        experiments: A list of experiment names to be processed.
        n_environments: The number of environments to process.
        path: The directory path where the CSV and JSON files will be saved.
            (Defaults to "data").
        data_range: A tuple specifying the range of data indices to be processed.
            If None, the entire length of the data is used. (Defaults to None).
        samplerate: The rate at which data points are sampled and saved to the CSV
            files. (Defaults to 1000).
    """
    filepath = Path(path)
    filepath.mkdir(parents=True, exist_ok=True)
    meta_data = {e: dm.raw_data[e].meta_data for e in experiments}
    # Save metadata to JSON
    with open(f"{filepath}.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)
    if not data_range:
        data_range = (0, dm.raw_data[experiments[0]].data.shape[-1])
    means = np.zeros((data_range[1] - data_range[0], len(experiments)))
    stds = np.zeros((data_range[1] - data_range[0], len(experiments)))
    column_names = ["idx"] + experiments + [f"{e}_std" for e in experiments]
    # Store means and standard deviations for each environment
    for env in range(n_environments):
        print(env)
        for i, name in enumerate(experiments):
            print(f"   {name}")
            sample = dm.raw_data[name]
            # Calculate means and standard deviations
            means[:, i] = np.mean(sample.data[env, :], axis=0)[
                data_range[0] : data_range[1]
            ]
            stds[:, i] = np.std(sample.data[env, :], axis=0)[
                data_range[0] : data_range[1]
            ]
        # Use 'column_name' header and write data to CSV file
        with open(f"{filepath}/random_{env + 1}.csv", "w", newline="") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(column_names)
            for j in range(data_range[0], data_range[1], samplerate):
                writer.writerow([str(j)] + [str(mean) for mean in means[j]])
            print(f"write: {filepath}/random_{i + 1}.csv")


save_random_as_csv(
    dm,
    ["q", "dq", "mmq", "avq", "redq", "vrq", "rraq_rho50_n10"],
    20,
    "data/random_environment",
    (0, 600000),
)


data_q = dm.raw_data["q"]
data_dq = dm.raw_data["dq"]
data_mmq = dm.raw_data["mmq"]
data_rq = dm.raw_data["rraq_rho50_n10"]
data_redq = dm.raw_data["redq"]
data_avq = dm.raw_data["avq"]
data_vrq = dm.raw_data["vrq"]

n = np.arange(data_dq.data.shape[-1]) + 1

fig, ax = plt.subplots(2, 3, figsize=(40, 20))
for i in range(2):
    for j in range(3):
        idx = (2 * i) + j
        ax[i, j].set_yscale("log")
        ax[i, j].ticklabel_format(style="sci", axis="x", scilimits=(5, 5))
        ax[i, j].plot(np.mean(data_q.data[idx], axis=0) * n, color="blue")
        ax[i, j].plot(np.mean(data_dq.data[idx], axis=0) * n, color="orange")
        ax[i, j].plot(np.mean(data_mmq.data[idx], axis=0) * n, color="green")
        ax[i, j].plot(np.mean(data_rq.data[idx], axis=0) * n, color="red")
        ax[i, j].plot(np.mean(data_avq.data[idx], axis=0) * n, color="purple")
        ax[i, j].plot(np.mean(data_vrq.data[idx], axis=0) * n, color="saddlebrown")
plt.savefig("random_environment_6.svg")
plt.savefig("random_environment_6.pdf")
plt.show()

fig, ax = plt.subplots(5, 4, figsize=(40, 20))
for i in range(5):
    for j in range(4):
        idx = (2 * i) + j
        ax[i, j].set_yscale("log")
        ax[i, j].ticklabel_format(style="sci", axis="x", scilimits=(5, 5))
        ax[i, j].plot(np.mean(data_q.data[idx], axis=0) * n, color="blue")
        ax[i, j].plot(np.mean(data_dq.data[idx], axis=0) * n, color="orange")
        ax[i, j].plot(np.mean(data_mmq.data[idx], axis=0) * n, color="green")
        ax[i, j].plot(np.mean(data_rq.data[idx], axis=0) * n, color="red")
        ax[i, j].plot(np.mean(data_avq.data[idx], axis=0) * n, color="purple")
        ax[i, j].plot(np.mean(data_vrq.data[idx], axis=0) * n, color="saddlebrown")
plt.savefig("random_environment_21.svg")
plt.savefig("random_environment_21.pdf")
plt.show()
