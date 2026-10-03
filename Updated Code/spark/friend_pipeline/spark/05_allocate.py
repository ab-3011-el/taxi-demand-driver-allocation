"""Batch driver allocation for all 168 weekly slots (defaults). The dashboard recomputes live with sliders."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from pyspark.sql import SparkSession

import config as C
from allocation import allocate, slot_frame

spark = SparkSession.builder.appName("05_allocate").getOrCreate()
read = lambda name: spark.read.parquet(f"{C.RESULTS}/{name}").toPandas()  # small tables (<50k rows)
grid, zones, supply = read("pred_grid"), read("zones"), read("supply_proxy")

fleet = int(os.environ.get("FLEET", C.DEFAULT_FLEET))
tpd = float(os.environ.get("TRIPS_PER_DRIVER", C.DEFAULT_TRIPS_PER_DRIVER))

zone_rows, move_rows = [], []
for dow in range(1, 8):
    for hour in range(24):
        z, mv = allocate(slot_frame(grid, zones, supply, dow, hour), fleet, tpd)
        zone_rows.append(z.assign(dow=dow, hour=hour)[
            ["dow", "hour", "zone_id", "prediction", "needed", "current", "assigned", "gap", "short_now", "short_after"]])
        move_rows.append(mv.assign(dow=dow, hour=hour))

zone_df = pd.concat(zone_rows, ignore_index=True).astype({"dow": "int32", "hour": "int32"})
move_df = pd.concat(move_rows, ignore_index=True).astype({"dow": "int32", "hour": "int32"})
spark.createDataFrame(zone_df).write.mode("overwrite").parquet(f"{C.RESULTS}/allocation")
spark.createDataFrame(move_df, "from_zone long, to_zone long, drivers long, km double, dow int, hour int") \
    .write.mode("overwrite").parquet(f"{C.RESULTS}/allocation_moves")
print(f"Allocation saved (fleet={fleet}, trips/driver/hr={tpd}): {len(zone_df):,} zone rows, {len(move_df):,} moves")
spark.stop()
