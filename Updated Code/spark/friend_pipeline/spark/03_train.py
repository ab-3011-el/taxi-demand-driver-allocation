"""Train GBT on Jan-May, test on June, compare against simple baselines, save model to HDFS."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession, functions as F

import config as C

spark = SparkSession.builder.appName("03_train").getOrCreate()
feat = spark.read.parquet(C.FEATURES)
is_train = F.col("date") < F.lit(C.TRAIN_END).cast("date")
train, test = feat.filter(is_train), feat.filter(~is_train)
print(f"Train rows: {train.count():,} | test rows: {test.count():,}")

pipe = Pipeline(stages=[
    VectorAssembler(inputCols=C.FEATURE_COLS, outputCol="features"),
    GBTRegressor(featuresCol="features", labelCol="demand", maxIter=60, maxDepth=6,
                 stepSize=0.1, subsamplingRate=0.8, seed=42),
])
model = pipe.fit(train)
model.write().overwrite().save(C.MODEL)

pred = (model.transform(test)
        .withColumn("prediction", F.greatest(F.col("prediction"), F.lit(0.0)))
        .withColumn("lag_168", F.col("lag_168").cast("double")).cache())


def score(name, col):
    r = pred.agg(F.avg(F.abs(F.col("demand") - F.col(col))).alias("mae"),
                 F.sqrt(F.avg((F.col("demand") - F.col(col)) ** 2)).alias("rmse")).first()
    return name, float(r.mae), float(r.rmse)


rows = [score("gbt_model", "prediction"),
        score("baseline_zone_hour_dow_mean", "hist_mean"),
        score("baseline_same_slot_last_week", "lag_168")]
spark.createDataFrame(rows, ["model", "mae", "rmse"]).write.mode("overwrite").parquet(f"{C.RESULTS}/metrics")
pred.select("zone_id", "ts", "demand", "prediction", "hist_mean").write.mode("overwrite").parquet(f"{C.RESULTS}/test_preds")

print(f"\n{'model':32s}{'MAE':>8s}{'RMSE':>8s}")
for n, mae, rmse in rows:
    print(f"{n:32s}{mae:8.3f}{rmse:8.3f}")
gain = 1 - rows[0][1] / rows[1][1]
print(f"\nMAE change vs zone-hour-dow baseline: {gain:+.1%}",
      "-> model adds value" if gain > 0 else "-> model does NOT beat the baseline")
spark.stop()
