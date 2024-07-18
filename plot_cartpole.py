from __future__ import annotations

import json
from pathlib import Path

from absl import logging
import matplotlib.pyplot as plt
import numpy as np

from agents.data_manager import DataManager

logging.set_verbosity(logging.DEBUG)

dm = DataManager()
dm.load_data("data/cartpole")


def save_cartpole_as_csv(
    dm: DataManager,
    experiments: list[str],
    path: str = "data",
) -> None:
    """Saves CartPole experiment data to a CSV file with accompanying metadata
    in a JSON file.

    This function collects hit times data from the specified experiments in the
    data manager and saves these values in a CSV file. The metadata for each
    experiment is saved in a JSON file.

    Args:
        dm: The data manager instance containing the raw data.
        experiments: A list of experiment names to be processed.
        path: The directory path where the CSV and JSON files will be saved.
            (Defaults to "data").
    """
    filepath = Path(path)
    filepath.mkdir(parents=True, exist_ok=True)
    hit_times = []
    meta_data = {}
    for _, name in enumerate(experiments):
        sample = dm.raw_data[name]
        hit_times.append(sample.data)
        meta_data[name] = sample.meta_data
    # Save hit times to CSV
    np.savetxt(
        f"{filepath}.csv",
        np.array(hit_times).T,
        delimiter=",",
        header=",".join(experiments),
        comments="",
    )
    # Save metadata to JSON
    with open(f"{filepath}.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)


save_cartpole_as_csv(dm, ["q", "dq", "mmq", "rraq", "avq"], "data/csv/cartpole")


for name, data in dm.raw_data.items():
    print(f"{name}: mean: {np.mean(data.data)} --std: {np.std(data.data)}")


fig, ax = plt.subplots(1, 5, sharex="col", sharey="row", figsize=(50, 10))
for i, (name, data) in enumerate(dm.raw_data.items()):
    ax[i].hist(data.data, 20, label=name)
    ax[i].set_xlim(0, 1000)
    ax[i].legend()
plt.savefig(
    "cartpole.svg",
    bbox_inches="tight",
    transparent="True",
    pad_inches=0,
)
plt.savefig(
    "cartpole.pdf",
    bbox_inches="tight",
    transparent="True",
    pad_inches=0,
)
plt.show()
