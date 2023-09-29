from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agents.data_manager import DataManager

dm = DataManager()
dm.load_data("data/lunarlander")


def save_lunarlander_as_csv(
    dm: DataManager,
    experiments: list[str],
    path: str = "data",
) -> None:
    """soon"""
    filepath = Path(path)
    filepath.mkdir(parents=True, exist_ok=True)
    hit_times = []
    meta_data = {}
    for _, name in enumerate(experiments):
        sample = dm.raw_data[name]
        hit_times.append(sample.data)
        meta_data[name] = sample.meta_data
    np.savetxt(
        f"{filepath}.csv",
        np.array(hit_times).T,
        delimiter=",",
        header=",".join(experiments),
        comments="",
    )
    with open(f"{filepath}.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)


save_lunarlander_as_csv(dm, ["q", "dq", "mmq", "rraq_rho25"], "data/csv/lunarlander")


experiments = ["q", "dq", "mmq", "rraq_rho25", "rraq_rho40"]
for e in experiments:
    print(
        f"{e}: mean {np.mean(dm.raw_data[e].data)} "
        f"std {np.std(dm.raw_data[e].data)}"
    )

experiments = ["q", "dq", "mmq", "rraq_rho25", "rraq_rho40"]
for e in experiments:
    print(f"{e}:\n{dm.raw_data[e].data}")
