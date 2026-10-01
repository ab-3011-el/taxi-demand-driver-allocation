from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("TaxiDemandFirstJob") \
    .getOrCreate()

print("\n======================================")
print("     TAXI DEMAND - FIRST SPARK JOB")
print("======================================\n")

# Read taxi CSV from HDFS
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs:///rides/raw/train.csv")

# Show schema
print("\n========== DATA SCHEMA ==========\n")
df.printSchema()

# Show first 10 records
print("\n========== FIRST 10 RECORDS ==========\n")
df.show(10, truncate=False)

# Count records
print("\n========== TOTAL RECORDS ==========\n")

total_records = df.count()

print(f"Total taxi trips: {total_records:,}")

print("\n======================================")
print("       SPARK JOB COMPLETED")
print("======================================\n")

spark.stop()
