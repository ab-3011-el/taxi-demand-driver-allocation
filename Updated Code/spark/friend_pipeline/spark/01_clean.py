"""Raw CSV (HDFS) -> cleaned parquet partitioned by date + zone lookup table."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyspark.sql import SparkSession, functions as F

import config as C

spark = SparkSession.builder.appName("01_clean").getOrCreate()
raw = spark.read.option("header", "true").option("inferSchema", "true").csv(C.RAW)


def in_box(lon, lat):
    return F.col(lon).between(C.LON_MIN, C.LON_MAX) & F.col(lat).between(C.LAT_MIN, C.LAT_MAX)


def zone(lon, lat):
    x = F.floor((F.col(lon) - C.LON_MIN) / C.CELL)
    y = F.floor((F.col(lat) - C.LAT_MIN) / C.CELL)
    return x * 100 + y


df = (raw
      .withColumn("pickup_ts", F.to_timestamp("pickup_datetime"))
      .withColumn("dropoff_ts", F.to_timestamp("dropoff_datetime"))
      .filter(in_box("pickup_longitude", "pickup_latitude") & in_box("dropoff_longitude", "dropoff_latitude"))
      .filter(F.col("trip_duration").between(60, 10800) & (F.col("passenger_count") > 0)
              & F.col("pickup_ts").isNotNull() & F.col("dropoff_ts").isNotNull())
      .withColumn("zone_id", zone("pickup_longitude", "pickup_latitude"))
      .withColumn("drop_zone_id", zone("dropoff_longitude", "dropoff_latitude"))
      .withColumn("date", F.to_date("pickup_ts"))
      .withColumn("hour", F.hour("pickup_ts"))
      .select("id", "pickup_ts", "dropoff_ts", "date", "hour", "zone_id", "drop_zone_id",
              "trip_duration", "passenger_count"))

df.write.mode("overwrite").partitionBy("date").parquet(C.CLEAN)
clean = spark.read.parquet(C.CLEAN)
print(f"Raw rows: {raw.count():,} | clean rows: {clean.count():,}")

# Top zones by pickup volume (+ cell centre for maps)
zones = (clean.groupBy("zone_id")
         .agg(F.count("*").alias("total_trips"), F.round(F.avg(F.col("trip_duration") / 60), 2).alias("avg_trip_min"))
         .orderBy(F.desc("total_trips")).limit(C.TOP_ZONES)
         .withColumn("lon", C.LON_MIN + (F.floor(F.col("zone_id") / 100) + 0.5) * C.CELL)
         .withColumn("lat", C.LAT_MIN + ((F.col("zone_id") % 100) + 0.5) * C.CELL))
zones.write.mode("overwrite").parquet(f"{C.RESULTS}/zones")
print(f"Zones kept: {zones.count()}")
spark.stop()
