from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BaselineResult:
    schedule: pd.DataFrame
    summary: dict


def run_baseline_dumb(
    vehicles: pd.DataFrame,
    site: pd.DataFrame,
    dt_minutes: int = 15,
    demand_charge_per_kw: float = 15.0,
    respect_site_limit: bool = False,
    allocation: str = "proportional",  # or "earliest_deadline"
) -> BaselineResult:
    """
    Dumb charging baseline:
      - Every plugged-in vehicle charges immediately at pmax until it hits target SOC.
    """
    dt_hours = dt_minutes / 60.0

    vids = vehicles["vehicle_id"].tolist()
    n = len(vids)
    T = len(site)

    building = site["building_load_kw"].to_numpy(float)
    contract = site["contract_limit_kw"].to_numpy(float)
    price = site["tou_price_per_kwh"].to_numpy(float)

    headroom = np.maximum(contract - building, 0.0)

    arr = vehicles["arrival_slot"].to_numpy(int)
    dep = vehicles["departure_slot"].to_numpy(int)
    pmax = vehicles["pmax_kw"].to_numpy(float)
    soc0 = vehicles["initial_soc"].to_numpy(float)
    soct = vehicles["target_soc"].to_numpy(float)
    cap = vehicles["battery_capacity_kwh"].to_numpy(float)
    eta = vehicles["charging_efficiency_eta"].to_numpy(float)

    # Track SOC over time
    soc = soc0.copy()
    P = np.zeros((n, T), dtype=float)

    for t in range(T):
        active = []
        desired = np.zeros(n, dtype=float)

        for i in range(n):
            if not (arr[i] <= t < dep[i]):
                continue

            remaining_soc = max(0.0, soct[i] - soc[i])
            if remaining_soc <= 1e-12:
                continue

            # battery energy needed (kWh)
            remaining_batt_kwh = remaining_soc * cap[i]
            # grid energy needed (kWh), accounting for efficiency
            remaining_grid_kwh = remaining_batt_kwh / max(eta[i], 1e-9)

            # maximum grid power this slot to exactly hit target
            p_to_finish = remaining_grid_kwh / dt_hours
            desired[i] = min(pmax[i], p_to_finish)
            active.append(i)

        if not active:
            continue

        if not respect_site_limit:
            # Truly dumb: ignore site limit
            P[:, t] = desired
        else:
            avail = headroom[t]
            total_desired = float(desired.sum())

            if avail <= 1e-12 or total_desired <= 1e-12:
                continue

            if total_desired <= avail + 1e-9:
                P[:, t] = desired
            else:
                if allocation == "proportional":
                    scale = avail / total_desired
                    P[:, t] = desired * scale
                elif allocation == "earliest_deadline":
                    # Give power to earliest departures first
                    order = sorted(active, key=lambda i: dep[i])
                    remaining = avail
                    for i in order:
                        take = min(desired[i], remaining)
                        P[i, t] = take
                        remaining -= take
                        if remaining <= 1e-9:
                            break
                else:
                    raise ValueError(f"Unknown allocation={allocation}")

        # Update SOC after applying P[:,t]
        # soc += (eta * P * dt) / cap
        soc += (eta * P[:, t] * dt_hours) / cap
        soc = np.minimum(soc, soct)  # cap at target for cleanliness

    # Build schedule dataframe 
    out = pd.DataFrame({"time_slot": np.arange(T)})
    for i, vid in enumerate(vids):
        out[f"{vid}_kw"] = P[i, :]

    ev_cols = [c for c in out.columns if c.endswith("_kw")]
    out["ev_total_kw"] = out[ev_cols].sum(axis=1)
    out["building_load_kw"] = building
    out["contract_limit_kw"] = contract
    out["site_total_kw"] = out["ev_total_kw"] + out["building_load_kw"]
    out["tou_price_per_kwh"] = price

    # SOC traces (end-of-slot)
    for i, vid in enumerate(vids):
        soc_trace = np.zeros(T + 1, dtype=float)
        soc_trace[0] = soc0[i]
        soc_trace[1:] = soc_trace[0] + np.cumsum(P[i, :] * dt_hours * eta[i] / cap[i])
        out[f"{vid}_soc"] = np.minimum(soc_trace[1:], 1.0)

    # Metrics
    energy_cost = float(np.sum(out["ev_total_kw"].to_numpy() * dt_hours * price))
    peak_ev = float(out["ev_total_kw"].max())
    peak_site = float(out["site_total_kw"].max())
    demand_cost = float(demand_charge_per_kw * peak_site)
    total_cost = float(energy_cost + demand_cost)

    # Violations only meaningful when respect_site_limit=False
    violation = out["site_total_kw"] - out["contract_limit_kw"]
    violation_pos = np.maximum(violation.to_numpy(float), 0.0)
    contract_violation_kw = float(violation_pos.max())
    violation_slots = int((violation_pos > 1e-6).sum())

    # Deadline misses: check SOC at end of (dep-1)
    misses = []
    for i, vid in enumerate(vids):
        t_check = int(dep[i]) - 1
        if 0 <= t_check < T:
            soc_at_dep = float(out.loc[t_check, f"{vid}_soc"])
            if soc_at_dep + 1e-6 < float(soct[i]):
                misses.append((vid, soc_at_dep, float(soct[i])))

    summary = {
        "dt_minutes": dt_minutes,
        "respect_site_limit": respect_site_limit,
        "allocation": allocation if respect_site_limit else None,
        "energy_cost_$": energy_cost,
        "peak_ev_kw": peak_ev,
        "peak_site_kw": peak_site,
        "max_contract_violation_kw": contract_violation_kw,
        "contract_violation_slots": violation_slots,
        "missed_deadlines": len(misses),
        "miss_details": misses[:5],  # small preview
        "demand_charge_per_kw_$": float(demand_charge_per_kw),
        "demand_cost_$": demand_cost,
        "total_cost_$": total_cost,

    }

    return BaselineResult(schedule=out, summary=summary)

