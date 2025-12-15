from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp

from .data_model import repo_root, load_vehicles_csv, load_site_csv


@dataclass
class SolveResult:
    feasible: bool
    schedule: pd.DataFrame
    summary: dict


def solve_lp_energy_only(
    vehicles: pd.DataFrame,
    site: pd.DataFrame,
    dt_minutes: int = 15,
    peak_weight: float = 1.0,
    solver_name: str | None = None,
) -> SolveResult:
    """
    Minimize energy cost (TOU) subject to:
      - charging only in [arrival, departure)
      - per-vehicle max power
      - site constraint: EV_total + building <= contract_limit
      - readiness by departure: SOC reaches target (via energy delivered with efficiency)
    """
    dt_hours = dt_minutes / 60.0

    vids = vehicles["vehicle_id"].tolist()
    n = len(vids)
    T = len(site)

    # Site series
    building = site["building_load_kw"].to_numpy(float)
    contract = site["contract_limit_kw"].to_numpy(float)
    price = site["tou_price_per_kwh"].to_numpy(float)

    site_available_for_evs = np.maximum(contract - building, 0.0)

    # Vehicle parameters
    arr = vehicles["arrival_slot"].to_numpy(int)
    dep = vehicles["departure_slot"].to_numpy(int)
    pmax = vehicles["pmax_kw"].to_numpy(float)
    soc0 = vehicles["initial_soc"].to_numpy(float)
    soct = vehicles["target_soc"].to_numpy(float)
    cap = vehicles["battery_capacity_kwh"].to_numpy(float)
    eta = vehicles["charging_efficiency_eta"].to_numpy(float)

    # Required battery energy (kWh)
    req_batt_kwh = np.maximum(0.0, (soct - soc0) * cap)

    # Decision variables: grid power per vehicle per slot (kW)
    P = cp.Variable((n, T), nonneg=True)

    constraints = []

    # Mask to force P=0 outside availability window and P<=pmax inside
    mask = np.zeros((n, T), dtype=float)
    for i in range(n):
        a = max(0, min(T, int(arr[i])))
        d = max(0, min(T, int(dep[i])))
        if d > a:
            mask[i, a:d] = 1.0
    constraints.append(P <= (pmax[:, None] * mask))

    # Site constraint per time slot
    constraints.append(cp.sum(P, axis=0) <= site_available_for_evs)

    # Readiness constraint: delivered battery energy >= required
    # Delivered battery energy per slot = eta_i * P[i,t] * dt_hours
    for i in range(n):
        if req_batt_kwh[i] <= 1e-9:
            continue
        a = max(0, min(T, int(arr[i])))
        d = max(0, min(T, int(dep[i])))
        constraints.append(cp.sum(P[i, a:d]) * dt_hours * eta[i] >= req_batt_kwh[i])

    # Objective: minimize TOU energy cost
    ev_total_kw = cp.sum(P, axis=0)
    peak_ev = cp.Variable(nonneg=True)
    constraints.append(ev_total_kw <= peak_ev)
    energy_cost = cp.sum(cp.multiply(ev_total_kw * dt_hours, price))
    obj = cp.Minimize(energy_cost + peak_weight * peak_ev)
    prob = cp.Problem(obj, constraints)

    # Solver selection
    if solver_name is None:
        solver_name = "HIGHS" if "HIGHS" in cp.installed_solvers() else "ECOS"

    try:
        prob.solve(solver=getattr(cp, solver_name), verbose=False)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=False)
        solver_name = "ECOS"

    feasible = prob.status in ("optimal", "optimal_inaccurate")

    # Build output schedule DF
    p_val = np.zeros((n, T))
    if feasible and P.value is not None:
        p_val = np.maximum(P.value, 0.0)

    out = pd.DataFrame({"time_slot": np.arange(T)})
    for i, vid in enumerate(vids):
        out[f"{vid}_kw"] = p_val[i, :]

    ev_cols = [c for c in out.columns if c.endswith("_kw")]
    out["ev_total_kw"] = out[ev_cols].sum(axis=1)
    out["building_load_kw"] = building
    out["contract_limit_kw"] = contract
    out["site_total_kw"] = out["ev_total_kw"] + out["building_load_kw"]
    out["tou_price_per_kwh"] = price

    # Add SOC traces (nice for debug + later reporting)
    for i, vid in enumerate(vids):
        soc = np.zeros(T + 1, dtype=float)
        soc[0] = soc0[i]
        # soc[t+1] = soc[t] + (eta * P * dt) / cap
        soc[1:] = soc[0] + np.cumsum(p_val[i, :] * dt_hours * eta[i] / cap[i])
        out[f"{vid}_soc"] = soc[1:]  # align SOC to end of slot

    realized_energy_cost = float(np.sum(out["ev_total_kw"].to_numpy() * dt_hours * price))
    summary = {
        "status": prob.status,
        "solver": solver_name,
        "dt_minutes": dt_minutes,
        "energy_cost_$": realized_energy_cost,
        "peak_ev_kw": float(out["ev_total_kw"].max()),
        "peak_site_kw": float(out["site_total_kw"].max()),
        "feasible": feasible,
        "peak_ev_kw_model": float(peak_ev.value) if feasible and peak_ev.value is not None else None,
        "peak_weight": peak_weight,
    }

    return SolveResult(feasible=feasible, schedule=out, summary=summary)


def _find_site_file(root: Path) -> Path:
    candidates = [
        root / "data" / "site_loads_and_tarrifs.csv",   # your current filename
        root / "data" / "site_loads_and_tariffs.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find site CSV. Tried: {[str(c) for c in candidates]}")


def main():
    root = repo_root()
    vehicles_path = root / "data" / "vehicles.csv"
    site_path = _find_site_file(root)

    vehicles = load_vehicles_csv(vehicles_path)
    site_series = load_site_csv(site_path, T=96)

    site_df = pd.DataFrame({
        "time_slot": site_series.slots,
        "timestamp_start": site_series.timestamp_start.values,
        "building_load_kw": site_series.building_load_kw.values,
        "tou_price_per_kwh": site_series.tou_price_per_kwh.values,
        "demand_charge_flag": site_series.demand_charge_flag.values,
        "contract_limit_kw": site_series.contract_limit_kw.values,
    })

    res = solve_lp_energy_only(vehicles, site_df, dt_minutes=15)

    print("\n=== Solve Summary ===")
    for k, v in res.summary.items():
        print(f"{k}: {v}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    res.schedule.to_csv(out_dir / "schedule.csv", index=False)
    print(f"\nSaved: {out_dir / 'schedule.csv'}")

    # Plot
    try:
        from .report_generator import plot_site_power
        plot_site_power(res.schedule, out_dir)
        print(f"Saved: {out_dir / 'site_power.png'}")
    except Exception as e:
        print(f"(Plot skipped) {e}")


if __name__ == "__main__":
    main()
