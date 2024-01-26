from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from absl import logging

from agents.data_manager import DataManager

logging.set_verbosity(logging.DEBUG)

dm = DataManager()
dm.load_data("data/cartpole")


def save_cartpole_as_csv(
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


save_cartpole_as_csv(dm, ["q", "dq", "mmq", "rraq"], "data/csv/cartpole")

steps_q = dm.raw_data["q"].data
steps_dq = dm.raw_data["dq"].data
steps_mmq = dm.raw_data["mmq"].data
steps_rq = dm.raw_data["rraq"].data


print(f"Q: mean={np.mean(steps_q)} & std={np.std(steps_q)}")
print(f"DQ: mean={np.mean(steps_dq)} & std={np.std(steps_dq)}")
print(f"MMQ: mean={np.mean(steps_mmq)} & std={np.std(steps_mmq)}")
print(f"RQ: mean={np.mean(steps_rq)} & std={np.std(steps_rq)}")


fig, (ax_q, ax_dq, ax_mmq, ax_rq) = plt.subplots(
    1, 4, sharex="col", sharey="row", figsize=(40, 10)
)
ax_q.hist(steps_q, 20, fc=(1, 0, 0, 1), label="Q")
ax_q.set_xlim(0, 1000)
ax_q.legend()
ax_dq.hist(steps_dq, 20, fc=(0, 1, 0, 1), label="DQ")
ax_dq.set_xlim(0, 1000)
ax_dq.legend()
ax_mmq.hist(steps_mmq, 20, fc=(0, 0, 1, 1), label="MMQ")
ax_mmq.set_xlim(0, 1000)
ax_mmq.legend()
ax_rq.hist(steps_rq, 20, fc=(0, 0, 0, 1), label="RQ")
ax_rq.set_xlim(0, 1000)
ax_rq.legend()
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


fig, ax = plt.subplots(1, 1, figsize=(40, 20))
ax.hist(steps_q, 20, fc=(1, 0, 0, 0.5), label="Q")
ax.hist(steps_dq, 20, fc=(0, 1, 0, 0.5), label="DQ")
ax.hist(steps_mmq, 20, fc=(0, 0, 1, 0.5), label="MMQ")
ax.hist(steps_rq, 20, fc=(0, 0, 0, 0.5), label="RQ")
ax.set_xlim(0, 1000)
ax.legend()
plt.savefig(
    "cartpole_shared.svg",
    bbox_inches="tight",
    transparent="True",
    pad_inches=0,
)
plt.savefig(
    "cartpole_shared.pdf",
    bbox_inches="tight",
    transparent="True",
    pad_inches=0,
)
plt.show()
