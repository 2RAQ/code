from __future__ import annotations

import csv
import json
from pathlib import Path

from absl import logging
import numpy as np

from agents.data_manager import DataManager

logging.set_verbosity(logging.DEBUG)

dm = DataManager()
dm.load_data("data/bairds_example")


def save_baird_as_csv(
    dm: DataManager,
    experiments: list[str],
    path: str = "data",
    data_range: None | tuple[int, int] = None,
) -> None:
    """Saves Baird's example data to a CSV file with accompanying metadata in a
    JSON file.

    Reads raw data from the passed DataManager instance and saves the mean and
    standard deviation in a CSV file. Additionally, the metadata for each
    experiment is saved in a JSON file.

    Args:
        dm: The data manager instance containing the raw data.
        experiments: A list of experiment names to be processed.
        path: The directory path where the CSV and JSON files will be saved.
            (Defaults to "data").
        data_range: A tuple specifying the range of data indices to be processed.
            If None, the entire length of the data is used. (Defaults to None).
    """
    filepath = Path(path)
    filepath.mkdir(parents=True, exist_ok=True)
    meta_data = {}
    if not data_range:
        data_range = (0, dm.raw_data[experiments[0]].data.shape[-1])
    means = np.zeros((data_range[1] - data_range[0], len(experiments)))
    stds = np.zeros((data_range[1] - data_range[0], len(experiments)))
    # Calculate means and standard deviations for each experiment
    for i, name in enumerate(experiments):
        sample = dm.raw_data[name]
        means[:, i] = np.mean(sample.data, axis=0)[0, data_range[0] : data_range[1]]
        stds[:, i] = np.std(sample.data, axis=0)[0, data_range[0] : data_range[1]]
        meta_data[name] = sample.meta_data
    # Prepare header and write data to CSV file
    experiments = ["idx"] + experiments + [f"{e}_std" for e in experiments]
    with open(f"{filepath}.csv", "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(experiments)
        for i in range(0, data_range[1] - data_range[0], 100):
            writer.writerow(
                [str(i)] + [str(x) for x in means[i]] + [str(x) for x in stds[i]]
            )
    # Write metadata to JSON file
    with open(f"{filepath}.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)


save_baird_as_csv(
    dm,
    ["q", "dq", "mmq", "avq", "redq", "vrq", "rraq_rho05_n10"],
    "data/bairds/compare",
    (0, 200000),
)
save_baird_as_csv(
    dm,
    [
        "q",
        "rraq_rho0_n10",
        "rraq_rho05_n10",
        "rraq_rho2_n10",
        "rraq_rho5_n10",
        "rraq_rho10_n10",
        "rraq_rho50_n10",
    ],
    "data/bairds/rhos",
    (0, 20000),
)
save_baird_as_csv(
    dm,
    [
        "rraq_rho05_n1",
        "rraq_rho05_n2",
        "rraq_rho05_n5",
        "rraq_rho05_n10",
        "rraq_rho05_n20",
    ],
    "data/bairds/Ns_1",
    (0, 20000),
)
save_baird_as_csv(
    dm,
    [
        "rraq_rho05_n1",
        "rraq_rho05_n2",
        "rraq_rho05_n5",
        "rraq_rho05_n10",
        "rraq_rho05_n20",
    ],
    "data/bairds/Ns_2",
    (0, 20000),
)

dm.plot_mse(
    range=(0, 200000),
    plot_data=["q", "dq", "mmq", "avq", "redq", "vrq", "rraq_rho05_n10"],
    logscale=True,
    times_n=True,
    save_fig_as="baird_005_compare",
)

dm.plot_mse(
    range=(0, 200000),
    plot_data=[
        "q",
        "rraq_rho0_n10",
        "rraq_rho05_n10",
        "rraq_rho2_n10",
        "rraq_rho5_n10",
        "rraq_rho10_n10",
        "rraq_rho50_n10",
    ],
    logscale=True,
    times_n=True,
    save_fig_as="baird_005_rhos",
)

dm.plot_mse(
    range=(0, 20000),
    plot_data=[
        "rraq_rho05_n1",
        "rraq_rho05_n2",
        "rraq_rho05_n5",
        "rraq_rho05_n10",
        "rraq_rho05_n20",
    ],
    logscale=False,
    times_n=False,
    save_fig_as="baird_005_Ns_1",
)

dm.plot_mse(
    range=(50000, 70000),
    plot_data=[
        "rraq_rho05_n1",
        "rraq_rho05_n2",
        "rraq_rho05_n5",
        "rraq_rho05_n10",
        "rraq_rho05_n20",
    ],
    logscale=False,
    times_n=False,
    save_fig_as="baird_005_Ns_2",
)
