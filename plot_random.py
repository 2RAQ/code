from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from absl import logging

from agents.data_manager import DataManager

logging.set_verbosity(logging.DEBUG)

dm = DataManager()
dm.load_data("data/random_environment")


def save_random_as_csv(
    dm: DataManager,
    experiments: list[str],
    n_environments: int,
    path: str = "data",
    data_range: None | tuple[int, int] = None,
    samplerate: int = 1000,
) -> None:
    """soon"""
    filepath = Path(path)
    filepath.mkdir(parents=True, exist_ok=True)
    meta_data = {e: dm.raw_data[e].meta_data for e in experiments}
    with open(f"{filepath}.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)
    if not data_range:
        data_range = (0, dm.raw_data[experiments[0]].data.shape[-1])
    means = np.zeros((data_range[1] - data_range[0], len(experiments)))
    stds = np.zeros((data_range[1] - data_range[0], len(experiments)))
    experiments = ["idx"] + experiments + [f"{e}_std" for e in experiments]
    for env in range(n_environments):
        for i, name in enumerate(experiments):
            sample = dm.raw_data[name]
            means[:, i] = np.mean(sample.data, axis=0)[
                env, data_range[0] : data_range[1]
            ]
            stds[:, i] = np.std(sample.data, axis=0)[env, data_range[0] : data_range[1]]
            with open(f"{filepath}/random_{i + 1}.csv", "w", newline="") as file:
                writer = csv.writer(file, delimiter=",")
                writer.writerow(experiments)
                for j in range(data_range[0], data_range[1], samplerate):
                    writer.writerow([str(j)] + [str(mean) for mean in means[j]])


save_random_as_csv(
    dm, ["q", "dq", "mmq", "rq_rho50_n10"], 20, "data/random_environment", (0, 600000)
)


data_q = dm.raw_data["q"]
data_dq = dm.raw_data["dq"]
data_mmq = dm.raw_data["mmq"]
data_rq = dm.raw_data["rq_rho50_n10"]

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
plt.savefig("random_environment_6.svg")
plt.savefig("random_environment_6.pdf")
# plt.show()

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
plt.savefig("random_environment_21.svg")
plt.savefig("random_environment_21.pdf")
# plt.show()
