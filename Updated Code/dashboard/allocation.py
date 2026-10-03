"""Pure pandas allocation logic, used by spark/05_allocate.py and the dashboard."""
import numpy as np
import pandas as pd

from config import CELL

KM_PER_DEG_LON, KM_PER_DEG_LAT = 84.4, 111.2  # at NYC latitude


def prev_slot(dow, hour):
    """(dow, hour) of the previous hour. dow: 1=Sunday .. 7=Saturday."""
    return (7 if dow == 1 else dow - 1, 23) if hour == 0 else (dow, hour - 1)


def slot_frame(grid, zones, supply, dow, hour):
    """One row per zone for a (dow, hour): predicted demand + drivers freed in the previous hour."""
    pdow, phour = prev_slot(dow, hour)
    g = grid[(grid.dow == dow) & (grid.hour == hour)][["zone_id", "prediction", "hist_mean"]]
    s = supply[(supply.dow == pdow) & (supply.hour == phour)][["zone_id", "avg_dropoffs"]]
    d = (g.merge(zones[["zone_id", "lon", "lat"]], on="zone_id")
          .merge(s, on="zone_id", how="left").fillna({"avg_dropoffs": 0.0}))
    d["zone_x"], d["zone_y"] = d.zone_id // 100, d.zone_id % 100
    return d.reset_index(drop=True)


def _apportion(weights, total):
    """Integer split of `total` proportional to weights (largest remainder); sums exactly to total."""
    w = np.asarray(weights, dtype=float)
    total = int(total)
    if total <= 0 or w.sum() <= 0:
        return np.zeros(len(w), dtype=int)
    raw = total * w / w.sum()
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    if rem > 0:
        base[np.argsort(-(raw - base))[:rem]] += 1
    return base


def allocate(d, fleet, trips_per_driver):
    """Return (zone table, moves table).

    needed   = ceil(predicted trips / trips_per_driver)
    assigned = needed if the fleet covers everyone, else the fleet split proportionally to demand
    current  = fleet split by where drivers were just freed (previous-hour drop-offs)
    gap      = assigned - current   (>0 needs inflow, <0 has surplus)
    """
    d = d.copy()
    d["needed"] = np.ceil(d.prediction / trips_per_driver).astype(int)
    fleet = int(fleet)
    d["assigned"] = d.needed if fleet >= d.needed.sum() else _apportion(d.prediction, fleet)
    w = d.avg_dropoffs if d.avg_dropoffs.sum() > 0 else np.ones(len(d))
    d["current"] = _apportion(w, fleet)
    d["gap"] = d.assigned - d.current
    d["short_now"] = (d.needed - d.current).clip(lower=0)
    d["short_after"] = (d.needed - d.assigned).clip(lower=0)

    # Greedy rebalancing: each deficit zone (largest first) pulls from the nearest surplus zones.
    sur = {int(z): [int(x), int(y), int(-g)]
           for z, x, y, g in zip(d.zone_id, d.zone_x, d.zone_y, d.gap) if g < 0}
    deficits = sorted(((int(z), int(x), int(y), int(g))
                       for z, x, y, g in zip(d.zone_id, d.zone_x, d.zone_y, d.gap) if g > 0),
                      key=lambda t: -t[3])
    moves = []
    for z, x, y, need in deficits:
        for sz in sorted(sur, key=lambda s: (sur[s][0] - x) ** 2 + (sur[s][1] - y) ** 2):
            if need == 0:
                break
            if sur[sz][2] <= 0:
                continue
            take = min(need, sur[sz][2])
            km = np.hypot((sur[sz][0] - x) * CELL * KM_PER_DEG_LON, (sur[sz][1] - y) * CELL * KM_PER_DEG_LAT)
            moves.append((sz, z, take, round(float(km), 2)))
            sur[sz][2] -= take
            need -= take
    mv = pd.DataFrame(moves, columns=["from_zone", "to_zone", "drivers", "km"]).astype(
        {"from_zone": "int64", "to_zone": "int64", "drivers": "int64", "km": "float64"})
    return d, mv
