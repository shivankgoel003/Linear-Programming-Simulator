from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_site_power(schedule: pd.DataFrame, out_dir: Path) -> None:
    x = schedule["time_slot"]

    plt.figure()
    plt.plot(x, schedule["building_load_kw"], label="Building kW")
    plt.plot(x, schedule["ev_total_kw"], label="EV total kW")
    plt.plot(x, schedule["site_total_kw"], label="Site total kW")
    plt.plot(x, schedule["contract_limit_kw"], label="Contract limit kW")
    plt.xlabel("Time slot")
    plt.ylabel("kW")
    plt.title("Site Power (Building + EVs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "site_power.png", dpi=160)
    plt.close()
