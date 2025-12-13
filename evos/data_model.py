from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SiteSeries:
    slots: pd.Index
    timestamp_start: pd.Series
    building_load_kw: pd.Series
    tou_price_per_kwh: pd.Series
    demand_charge_flag: pd.Series
    contract_limit_kw: pd.Series


def load_vehicles_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "vehicle_id",
        "arrival_slot",
        "departure_slot",
        "initial_soc",
        "target_soc",
        "pmax_kw",
        "battery_capacity_kwh",
        "charging_efficiency_eta",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"vehicles.csv missing columns: {sorted(missing)}")

    df = df.copy()
    df["vehicle_id"] = df["vehicle_id"].astype(str)
    df["arrival_slot"] = df["arrival_slot"].astype(int)
    df["departure_slot"] = df["departure_slot"].astype(int)

    for c in ["initial_soc", "target_soc", "pmax_kw", "battery_capacity_kwh", "charging_efficiency_eta"]:
        df[c] = df[c].astype(float)

    # Basic validation
    if (df["departure_slot"] <= df["arrival_slot"]).any():
        bad = df[df["departure_slot"] <= df["arrival_slot"]][["vehicle_id", "arrival_slot", "departure_slot"]]
        raise ValueError(f"Some vehicles have departure_slot <= arrival_slot:\n{bad}")

    if ((df["initial_soc"] < 0) | (df["initial_soc"] > 1) | (df["target_soc"] < 0) | (df["target_soc"] > 1)).any():
        raise ValueError("initial_soc/target_soc must be in [0, 1].")

    if (df["charging_efficiency_eta"] <= 0).any() or (df["battery_capacity_kwh"] <= 0).any():
        raise ValueError("charging_efficiency_eta and battery_capacity_kwh must be > 0.")

    if (df["pmax_kw"] <= 0).any():
        raise ValueError("pmax_kw must be > 0.")

    return df


def load_site_csv(path: Path, T: int = 96) -> SiteSeries:
    df = pd.read_csv(path)

    required = {
        "time_slot",
        "timestamp_start",
        "building_load_kw",
        "tou_price_per_kwh",
        "demand_charge_flag",
        "contract_limit_kw",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"site_loads_and_tarrifs.csv missing columns: {sorted(missing)}")

    df = df.copy()
    df["time_slot"] = df["time_slot"].astype(int)
    df = df.sort_values("time_slot").set_index("time_slot")

    # Reindex to 0..T-1; ffill/bfill allows you to start with partial rows too
    full_index = pd.Index(range(T), name="time_slot")
    df = df.reindex(full_index).ffill().bfill()

    return SiteSeries(
        slots=full_index,
        timestamp_start=df["timestamp_start"].astype(str),
        building_load_kw=df["building_load_kw"].astype(float),
        tou_price_per_kwh=df["tou_price_per_kwh"].astype(float),
        demand_charge_flag=df["demand_charge_flag"].astype(int),
        contract_limit_kw=df["contract_limit_kw"].astype(float),
    )

