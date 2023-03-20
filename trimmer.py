from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window

import json


import time

min_size = 0

#UDF to extract only the pressure values from the list.....
trim_array = F.udf(lambda row: list(map(float, row[:min_size])), T.ArrayType(T.FloatType(), containsNull=False))

# Create a SparkSession
spark = SparkSession \
    .builder \
    .appName("Data Stager") \
    .config("spark.driver.memory", '8g') \
    .getOrCreate()

spark.sparkContext.setLogLevel('ERROR')

# get the start time
st = time.time()

phm_df = spark.read.parquet("stage/")

phm_df = phm_df.withColumn('size_pdmp', F.size(F.col('pdmp')))
phm_df = phm_df.withColumn('size_pin', F.size(F.col('pin')))
phm_df = phm_df.withColumn('size_po', F.size(F.col('po')))

count = phm_df.count()

phm_df.select(F.min('size_pdmp'), F.max('size_pdmp'), F.min('size_pin'), F.max('size_pin'), F.min('size_po'), F.max('size_po')).show()
min_size = phm_df.select(F.min('size_pdmp')).collect()[0][0]

phm_df = phm_df.withColumn('trimmed_pdmp', trim_array(phm_df['pdmp']))
phm_df = phm_df.withColumn('trimmed_pin', trim_array(phm_df['pin']))
phm_df = phm_df.withColumn('trimmed_po', trim_array(phm_df['po']))
phm_df = phm_df.withColumn('trimmed_size_pdmp', F.size(F.col('trimmed_pdmp')))
phm_df = phm_df.withColumn('trimmed_size_pin', F.size(F.col('trimmed_pin')))
phm_df = phm_df.withColumn('trimmed_size_po', F.size(F.col('trimmed_po')))


phm_df = phm_df.drop('pdmp', 'pin', 'po', 'size_pdmp', 'size_pin', 'size_po', 'trimmed_size_pdmp', 'trimmed_size_pin', 'trimmed_size_po')

phm_df = phm_df.withColumnRenamed('trimmed_pdmp', 'pdmp')
phm_df = phm_df.withColumnRenamed('trimmed_pin', 'pin')
phm_df = phm_df.withColumnRenamed('trimmed_po', 'po')

count = phm_df.count()

phm_df = phm_df.repartition(5, 'individual')
phm_df.write.mode('overwrite').parquet("../output/")


# get the end time
et = time.time()


# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')