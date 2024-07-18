from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
import os
from pathlib import Path
import pickle
from typing import Any

from absl import logging
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DataContainer:
    """A container class to store data and metadata of experiments inside the
    DataManager.

    Attributes:
        name: A string to identify the data container.
        data: A numpy array to store the data.
        meta_data: A dictionary to store metadata of the data container.
    """

    name: str = ""
    data: np.ndarray = np.array([])
    meta_data: dict[str, Any] = field(default_factory=dict)

    def add_entry(self, key: str, content: Any) -> None:
        self.meta_data[key] = content


class DataManager:
    """A class to manage the data of experiments.

    The DataManager class can be used to store experiment results and their
    respective metadata. It allows to write the data to disk and load it back as
    well as to generate basic visualization of the stored data.
    """

    def __init__(self) -> None:
        self.raw_data: dict[str, DataContainer] = {}

    def add_raw_data(self, data: DataContainer) -> None:
        if not isinstance(data, DataContainer):  # type: ignore[unreachable]
            raise TypeError(
                f"Expected 'data' to be of type 'DataContainer'; "
                f"received 'data' of type: {type(data).__name__}"
            )
        container_name = data.name
        if len(container_name) == 0:
            container_name = f"unnamed_data_{len(self.raw_data)}"
        self.raw_data[container_name] = data
        logging.debug(
            f"Added {data.name} containing data of shape {data.data.shape} "
            f" with metadata: {data.meta_data}"
        )

    def get_data(self, key: str) -> DataContainer:
        return self.raw_data[key]

    def save_data(self, path: str = "data") -> None:
        """Pickles the data dictionary to disk"""
        filepath = Path(path)
        filepath.mkdir(parents=True, exist_ok=True)
        for name, data in self.raw_data.items():
            with open(filepath / name, "wb") as file:
                pickle.dump(data, file)

    def load_data(self, directory: str) -> None:
        """Loads pickled data, written by this class from disk into the
        data dictionary
        """
        for file in os.listdir(directory):
            logging.debug(f"Loading file: {file}")
            with open(f"{directory}/{file}", "rb") as f:
                self.add_raw_data(pickle.load(f))

    def plot_mse(
        self,
        range: None | tuple[int, int] = None,
        plot_data: None | list[str] = None,
        logscale: bool = True,
        times_n: bool = False,
        save_fig_as: None | str = None,
    ) -> None:
        """Plots the (asymptotic) mean squared error of the stored data.

        Args:
            range: A tuple of two integers to specify the range of steps to plot.
            plot_data: A list of strings to specify the data containers to plot.
            logscale: A boolean to specify if the y-axis should be logarithmic.
            times_n: A boolean to specify if the data values should be multiplied
                by n which results in the ASME.
            save_fig_as: A string to specify the filename to save the figure to,
                does not save the figure if None.
        """

        if not plot_data:
            plot_data = list(self.raw_data.keys())
        _, ax_mse = plt.subplots(1, figsize=(20, 10))
        if logscale:
            ax_mse.set_yscale("log")
        for data_id in plot_data:
            data = np.mean(self.raw_data[data_id].data, axis=1)
            mse = np.mean(data, axis=0)
            se = np.std(data, axis=0) / np.sqrt(data.shape[0])
            if not range:
                range = (0, mse.shape[0])
            n_samples = np.arange(data.shape[1])
            if times_n:
                mse *= n_samples + 1
            ax_mse.errorbar(
                n_samples[range[0] : range[1]],
                mse[range[0] : range[1]],
                se[range[0] : range[1]],
                capsize=2.5,
                errorevery=10000,
                markevery=10000,
                label=self._gen_label(self.raw_data[data_id].meta_data),
            )
        if times_n:
            ax_mse.set_ylabel("AMSE")
        else:
            ax_mse.set_ylabel("MSE")
        ax_mse.set_xlabel("Step")
        plt.legend()
        if save_fig_as:
            plt.savefig(
                f"{save_fig_as}.svg",
                bbox_inches="tight",
                transparent="True",
                pad_inches=0,
            )
            plt.savefig(
                f"{save_fig_as}.png",
                facecolor="white",
                bbox_inches="tight",
                transparent="True",
                pad_inches=0,
            )
        plt.show()

    def _gen_label(self, meta_data: dict[str, Any]) -> str:
        """Generates labels based on some predefined metadata keys"""
        label = f"{meta_data['name']} with N={meta_data['thetas']}"
        if "rho" in meta_data:
            label += f" & rho={meta_data['rho']}"
        if "K" in meta_data:
            label += f" & K={meta_data['K']}"
        if "M" in meta_data:
            label += f" & M={meta_data['M']}"
        if "D" in meta_data:
            label += f" & D={meta_data['D']}"
        return label
