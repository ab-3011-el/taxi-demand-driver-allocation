from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, to_timestamp
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

spark = SparkSession.builder \
    .appName("TaxiDemandPrediction") \
    .getOrCreate()

print("\n==========================================")
print("       TAXI DEMAND PREDICTION")
print("==========================================\n")

# --------------------------------------------------
# 1. Read taxi data from HDFS
# --------------------------------------------------

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs:///rides/raw/train.csv")

# --------------------------------------------------
# 2. Convert pickup time
# --------------------------------------------------

df = df.withColumn(
    "pickup_time",
    to_timestamp(col("pickup_datetime"))
)

# --------------------------------------------------
# 3. Create prediction features
# --------------------------------------------------

df = df.withColumn(
    "pickup_hour",
    hour(col("pickup_time"))
)

df = df.withColumn(
    "day_of_week",
    dayofweek(col("pickup_time"))
)

# --------------------------------------------------
# 4. Create hourly demand dataset
# --------------------------------------------------

demand = df.groupBy(
    "pickup_hour",
    "day_of_week"
).count()

demand = demand.withColumnRenamed(
    "count",
    "total_trips"
)

# --------------------------------------------------
# 5. Prepare ML features
# --------------------------------------------------

assembler = VectorAssembler(
    inputCols=[
        "pickup_hour",
        "day_of_week"
    ],
    outputCol="features"
)

data = assembler.transform(demand)

# --------------------------------------------------
# 6. Split training and testing data
# --------------------------------------------------

train_data, test_data = data.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Training records:", train_data.count())
print("Testing records:", test_data.count())

# --------------------------------------------------
# 7. Train Random Forest model
# --------------------------------------------------

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="total_trips",
    numTrees=50,
    seed=42
)

model = rf.fit(train_data)

# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

predictions = model.transform(test_data)

print("\n========== DEMAND PREDICTIONS ==========\n")

predictions.select(
    "pickup_hour",
    "day_of_week",
    "total_trips",
    "prediction"
).orderBy(
    "pickup_hour"
).show(
    20,
    truncate=False
)

# --------------------------------------------------
# 9. Save predictions to HDFS
# --------------------------------------------------

predictions.select(
    "pickup_hour",
    "day_of_week",
    "total_trips",
    "prediction"
).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("hdfs:///rides/results/demand_predictions")

print("\n==========================================")
print("       DEMAND PREDICTION COMPLETED")
print("==========================================\n")

spark.stop()
