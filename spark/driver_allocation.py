from pyspark.sql import SparkSession
from pyspark.sql.functions import col, ceil, lit, round, when

# ==========================================
# CREATE SPARK SESSION
# ==========================================

spark = SparkSession.builder \
    .appName("TaxiDriverAllocation") \
    .getOrCreate()

print("\n==========================================")
print("        TAXI DRIVER ALLOCATION")
print("==========================================\n")


# ==========================================
# 1. LOAD PREDICTED DEMAND
# ==========================================

predictions = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs:///rides/results/demand_predictions")

print("Predicted demand records:", predictions.count())


# ==========================================
# 2. DRIVER CAPACITY ASSUMPTION
# ==========================================
# Assumption:
# One driver can handle approximately
# 3 taxi trips per hour.

TRIPS_PER_DRIVER = 3


# ==========================================
# 3. CALCULATE REQUIRED DRIVERS
# ==========================================

allocation = predictions.withColumn(
    "required_drivers",
    ceil(
        col("prediction") / lit(TRIPS_PER_DRIVER)
    )
)


# ==========================================
# 4. SIMULATED DRIVER AVAILABILITY
# ==========================================
# The Kaggle dataset does not contain
# real-time driver availability.
#
# Therefore, we use a simulated supply
# of 3000 available drivers.

AVAILABLE_DRIVERS = 3000

allocation = allocation.withColumn(
    "available_drivers",
    lit(AVAILABLE_DRIVERS)
)


# ==========================================
# 5. CALCULATE DRIVER DIFFERENCE
# ==========================================

allocation = allocation.withColumn(
    "driver_difference",
    col("available_drivers") -
    col("required_drivers")
)


# ==========================================
# 6. DETERMINE ALLOCATION STATUS
# ==========================================

allocation = allocation.withColumn(
    "allocation_status",
    when(
        col("driver_difference") >= 0,
        "Sufficient Drivers"
    ).otherwise(
        "Driver Shortage"
    )
)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n========== DRIVER ALLOCATION RESULTS ==========\n")

allocation.select(
    "pickup_hour",
    "day_of_week",
    round(
        col("prediction"),
        2
    ).alias("predicted_demand"),
    "required_drivers",
    "available_drivers",
    "driver_difference",
    "allocation_status"
).orderBy(
    "pickup_hour",
    "day_of_week"
).show(
    30,
    truncate=False
)


# ==========================================
# 8. SAVE RESULTS TO HDFS
# ==========================================

allocation.select(
    "pickup_hour",
    "day_of_week",
    round(
        col("prediction"),
        2
    ).alias("predicted_demand"),
    "required_drivers",
    "available_drivers",
    "driver_difference",
    "allocation_status"
).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(
        "hdfs:///rides/results/driver_allocation"
    )


# ==========================================
# 9. COMPLETION MESSAGE
# ==========================================

print("\n==========================================")
print("       DRIVER ALLOCATION COMPLETED")
print("==========================================\n")


# ==========================================
# STOP SPARK
# ==========================================

spark.stop()
