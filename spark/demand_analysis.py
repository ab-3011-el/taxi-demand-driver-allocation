from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    hour,
    dayofweek,
    date_format,
    count,
    avg,
    round
)

# --------------------------------------------------
# Create Spark Session
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("TaxiDemandAnalysis") \
    .getOrCreate()

print("\n==========================================")
print("       TAXI DEMAND ANALYSIS")
print("==========================================\n")

# --------------------------------------------------
# Read taxi data from HDFS
# --------------------------------------------------

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs:///rides/raw/train.csv")

print("Total records loaded:", df.count())

# --------------------------------------------------
# Create useful time features
# --------------------------------------------------

df = df.withColumn(
    "pickup_hour",
    hour("pickup_datetime")
)

df = df.withColumn(
    "day_number",
    dayofweek("pickup_datetime")
)

df = df.withColumn(
    "day_name",
    date_format("pickup_datetime", "EEEE")
)

# --------------------------------------------------
# 1. Demand by Hour
# --------------------------------------------------

hourly_demand = df.groupBy(
    "pickup_hour"
).agg(
    count("*").alias("total_trips")
).orderBy(
    "pickup_hour"
)

print("\n========== DEMAND BY HOUR ==========\n")

hourly_demand.show(24, truncate=False)

# --------------------------------------------------
# 2. Demand by Day
# --------------------------------------------------

daily_demand = df.groupBy(
    "day_number",
    "day_name"
).agg(
    count("*").alias("total_trips")
).orderBy(
    "day_number"
)

print("\n========== DEMAND BY DAY ==========\n")

daily_demand.show(7, truncate=False)

# --------------------------------------------------
# 3. Average Trip Duration
# --------------------------------------------------

average_duration = df.select(
    round(
        avg("trip_duration") / 60,
        2
    ).alias("average_duration_minutes")
)

print("\n========== AVERAGE TRIP DURATION ==========\n")

average_duration.show()

# --------------------------------------------------
# 4. Average Passenger Count
# --------------------------------------------------

average_passengers = df.select(
    round(
        avg("passenger_count"),
        2
    ).alias("average_passengers")
)

print("\n========== AVERAGE PASSENGERS ==========\n")

average_passengers.show()

# --------------------------------------------------
# Save results to HDFS
# --------------------------------------------------

hourly_demand.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("hdfs:///rides/results/hourly_demand")

daily_demand.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("hdfs:///rides/results/daily_demand")

average_duration.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("hdfs:///rides/results/average_duration")

average_passengers.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("hdfs:///rides/results/average_passengers")

print("\n==========================================")
print("      DEMAND ANALYSIS COMPLETED")
print("==========================================\n")

spark.stop()
