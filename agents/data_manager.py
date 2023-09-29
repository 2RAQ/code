from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from absl import logging

if TYPE_CHECKING:
    pass


@dataclass
class DataContainer:
    """soon"""

    name: str = ""
    data: np.ndarray = np.array([])
    meta_data: dict[str, Any] = field(default_factory=dict)

    def add_entry(self, key: str, content: Any) -> None:
        """soon"""
        self.meta_data[key] = content


class DataManager:
    """soon"""

    def __init__(self) -> None:
        """soon"""
        self.raw_data: dict[str, DataContainer] = {}

    def add_raw_data(self, data: DataContainer) -> None:
        """soon"""
        if not isinstance(data, DataContainer):
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
        """soon"""
        return self.raw_data[key]

    def save_data(self, path: str = "data") -> None:
        """soon"""
        filepath = Path(path)
        filepath.mkdir(parents=True, exist_ok=True)
        for name, data in self.raw_data.items():
            with open(filepath / name, "wb") as file:
                pickle.dump(data, file)

    def load_data(self, directory: str) -> None:
        """Add method to selectively load data"""
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
        """soon"""

        if not plot_data:
            plot_data = list(self.raw_data.keys())
        _, ax_mse = plt.subplots(1, figsize=(40, 20))
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
                errorevery=500,
                markevery=500,
                label=self._gen_label(self.raw_data[data_id].meta_data),
            )
        if times_n:
            ax_mse.set_ylabel("nMSE")
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
        plt.show()

    def _gen_label(self, meta_data: dict[str, Any]) -> str:
        """soon"""
        label = f"{meta_data['name']} with N=" f"{meta_data['thetas']}"
        if meta_data["rho"]:
            label += f" & rho={meta_data['rho']}"
        return label
