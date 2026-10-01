from pyspark.sql import SparkSession
from pyspark.sql.functions import col, floor, count, round, avg

spark = SparkSession.builder \
    .appName("TaxiZoneAnalysis") \
    .getOrCreate()

print("\n==========================================")
print("          TAXI ZONE ANALYSIS")
print("==========================================\n")

# --------------------------------------------------
# 1. Read taxi data from HDFS
# --------------------------------------------------

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs:///rides/raw/train.csv")

total_records = df.count()

print("Total records loaded:", total_records)

# --------------------------------------------------
# 2. Keep only valid NYC coordinates
# --------------------------------------------------

df = df.filter(
    (col("pickup_longitude") >= -74.05) &
    (col("pickup_longitude") <= -73.70) &
    (col("pickup_latitude") >= 40.55) &
    (col("pickup_latitude") <= 40.95)
)

valid_records = df.count()

print("Valid NYC records:", valid_records)
print("Records removed:", total_records - valid_records)

# --------------------------------------------------
# 3. Create geographic grid zones
# --------------------------------------------------

df = df.withColumn(
    "zone_x",
    floor((col("pickup_longitude") + 74.05) / 0.05)
)

df = df.withColumn(
    "zone_y",
    floor((col("pickup_latitude") - 40.55) / 0.05)
)

df = df.withColumn(
    "zone_id",
    col("zone_x") * 100 + col("zone_y")
)

# --------------------------------------------------
# 4. Calculate zone demand
# --------------------------------------------------

zone_demand = df.groupBy(
    "zone_id"
).agg(
    count("*").alias("total_trips"),

    round(
        avg("trip_duration") / 60,
        2
    ).alias("avg_trip_duration_minutes"),

    round(
        avg("passenger_count"),
        2
    ).alias("avg_passengers")
).orderBy(
    col("total_trips").desc()
)

# --------------------------------------------------
# 5. Display top 20 zones
# --------------------------------------------------

print("\n========== TOP 20 HIGH-DEMAND ZONES ==========\n")

zone_demand.show(
    20,
    truncate=False
)

# --------------------------------------------------
# 6. Save results to HDFS
# --------------------------------------------------

zone_demand.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("hdfs:///rides/results/zone_demand")

print("\n==========================================")
print("       ZONE ANALYSIS COMPLETED")
print("==========================================\n")

spark.stop()
