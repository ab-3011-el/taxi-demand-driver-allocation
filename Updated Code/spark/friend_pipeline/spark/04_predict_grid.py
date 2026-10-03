"""Score every zone x day-of-week x hour once, so the dashboard needs no Spark at query time."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, functions as F

import config as C

spark = SparkSession.builder.appName("04_predict_grid").getOrCreate()
model = PipelineModel.load(C.MODEL)

# hist already holds every zone x dow x hour. For a "typical" slot, last week's value = the typical value.
grid = (spark.read.parquet(f"{C.RESULTS}/hist")
        .withColumn("is_weekend", F.col("dow").isin(1, 7).cast("int"))
        .withColumn("zone_x", F.floor(F.col("zone_id") / 100).cast("int"))
        .withColumn("zone_y", (F.col("zone_id") % 100).cast("int"))
        .withColumn("lag_168", F.col("hist_mean")))

out = (model.transform(grid)
       .select("zone_id", "dow", "hour", F.round("hist_mean", 2).alias("hist_mean"),
               F.round(F.greatest(F.col("prediction"), F.lit(0.0)), 2).alias("prediction")))
out.write.mode("overwrite").parquet(f"{C.RESULTS}/pred_grid")
print("pred_grid rows:", out.count())
spark.stop()
