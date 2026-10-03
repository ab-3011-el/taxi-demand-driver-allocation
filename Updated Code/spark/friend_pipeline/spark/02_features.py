"""Cleaned trips -> zone x hour demand table (zero-filled) with ML features + analysis tables."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyspark.sql import SparkSession, Window, functions as F

import config as C

spark = SparkSession.builder.appName("02_features").getOrCreate()
clean = spark.read.parquet(C.CLEAN)
zones = spark.read.parquet(f"{C.RESULTS}/zones").select("zone_id")

# Complete hourly timeline x zones, so quiet hours become explicit zeros
b = clean.agg(F.min("pickup_ts").alias("a"), F.max("pickup_ts").alias("b")).first()
fmt = "%Y-%m-%d %H:00:00"
hours = spark.sql(f"SELECT explode(sequence(to_timestamp('{b.a.strftime(fmt)}'), "
                  f"to_timestamp('{b.b.strftime(fmt)}'), interval 1 hour)) AS ts")
grid = zones.crossJoin(hours)

demand = (clean.join(zones, "zone_id")
          .groupBy("zone_id", F.date_trunc("hour", "pickup_ts").alias("ts"))
          .agg(F.count("*").alias("demand")))
# Supply proxy: drop-offs = drivers that just became free in that zone
dropoffs = (clean.join(zones.withColumnRenamed("zone_id", "drop_zone_id"), "drop_zone_id")
            .groupBy(F.col("drop_zone_id").alias("zone_id"), F.date_trunc("hour", "dropoff_ts").alias("ts"))
            .agg(F.count("*").alias("dropoffs")))

data = (grid.join(demand, ["zone_id", "ts"], "left").join(dropoffs, ["zone_id", "ts"], "left")
        .fillna(0, ["demand", "dropoffs"])
        .withColumn("date", F.to_date("ts")).withColumn("hour", F.hour("ts"))
        .withColumn("dow", F.dayofweek("ts"))  # 1=Sunday .. 7=Saturday
        .withColumn("is_weekend", F.col("dow").isin(1, 7).cast("int"))
        .withColumn("zone_x", F.floor(F.col("zone_id") / 100).cast("int"))
        .withColumn("zone_y", (F.col("zone_id") % 100).cast("int"))
        .withColumn("lag_168", F.lag("demand", 168).over(Window.partitionBy("zone_id").orderBy("ts")))
        ).cache()

# Historical mean per zone/dow/hour from TRAIN dates only. Training rows use leave-one-out
# (their own value excluded) so the model cannot just copy its target.
is_train = F.col("date") < F.lit(C.TRAIN_END).cast("date")
stats = (data.filter(is_train).groupBy("zone_id", "dow", "hour")
         .agg(F.sum("demand").alias("s"), F.count("*").alias("n")))
data = (data.join(stats, ["zone_id", "dow", "hour"], "left")
        .withColumn("hist_mean", F.when(is_train & (F.col("n") > 1), (F.col("s") - F.col("demand")) / (F.col("n") - 1))
                    .otherwise(F.col("s") / F.col("n"))))

(stats.withColumn("hist_mean", F.col("s") / F.col("n")).select("zone_id", "dow", "hour", "hist_mean")
 .write.mode("overwrite").parquet(f"{C.RESULTS}/hist"))
(data.groupBy("zone_id", "dow", "hour").agg(F.avg("dropoffs").alias("avg_dropoffs"))
 .write.mode("overwrite").parquet(f"{C.RESULTS}/supply_proxy"))
(data.groupBy("date", "dow", "hour").agg(F.sum("demand").alias("t"))
 .groupBy("dow", "hour").agg(F.round(F.avg("t"), 1).alias("avg_city_trips"))
 .write.mode("overwrite").parquet(f"{C.RESULTS}/demand_profile"))

(data.filter(F.col("lag_168").isNotNull())  # first week has no lag
 .select("zone_id", "ts", "date", *C.FEATURE_COLS, "demand", "dropoffs")
 .write.mode("overwrite").parquet(C.FEATURES))
print("Feature rows:", spark.read.parquet(C.FEATURES).count())
spark.stop()
