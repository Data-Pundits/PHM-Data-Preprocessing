"""
## INFO:
## This script requires the 'preprocessing.py' script to be run in prior and the 'stage/' folder should have some data.

## This script will find the least amount of values present in the 
## pdmp, pin and po array columns and limit them to that minimum value.
## The final processed dataframe is written to an 'output/' folder.

## This will ensure that no null values are passed while training classification models in the future steps.

## The next script to be run in sequence is the 'feature-extraction.py' script.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window

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

# Read the previously consolidated data from the 'consolidated_stage/' folder:
phm_df = spark.read.parquet("consolidated_stage/")

# Find the sizes of the 3 sensor columns:
phm_df = phm_df.withColumn('size_pdmp', F.size(F.col('pdmp')))
phm_df = phm_df.withColumn('size_pin', F.size(F.col('pin')))
phm_df = phm_df.withColumn('size_po', F.size(F.col('po')))

count = phm_df.count()

# Find the minimum and max sizes of these 3 sensor columns:
phm_df.select(F.min('size_pdmp'), F.max('size_pdmp'), F.min('size_pin'), F.max('size_pin'), F.min('size_po'), F.max('size_po')).show()
min_size = phm_df.select(F.min('size_pdmp')).collect()[0][0]

# Use the min-size value to trim the arrays containing more than the min-size number of values in the 3 sensor columns:
phm_df = phm_df.withColumn('trimmed_pdmp', trim_array(phm_df['pdmp']))
phm_df = phm_df.withColumn('trimmed_pin', trim_array(phm_df['pin']))
phm_df = phm_df.withColumn('trimmed_po', trim_array(phm_df['po']))
phm_df = phm_df.withColumn('trimmed_size_pdmp', F.size(F.col('trimmed_pdmp')))
phm_df = phm_df.withColumn('trimmed_size_pin', F.size(F.col('trimmed_pin')))
phm_df = phm_df.withColumn('trimmed_size_po', F.size(F.col('trimmed_po')))

# Drop intermediate columns that are no longer needed:
phm_df = phm_df.drop('pdmp', 'pin', 'po', 'size_pdmp', 'size_pin', 'size_po', 'trimmed_size_pdmp', 'trimmed_size_pin', 'trimmed_size_po')

# Rename the trimmed columns to its original names:
phm_df = phm_df.withColumnRenamed('trimmed_pdmp', 'pdmp')
phm_df = phm_df.withColumnRenamed('trimmed_pin', 'pin')
phm_df = phm_df.withColumnRenamed('trimmed_po', 'po')

count = phm_df.count()

# Repartition the new final dataframe and write to the 'output/' folder:
phm_df = phm_df.repartition(5, 'individual')
phm_df.write.mode('overwrite').parquet("consolidated_trimmed_stage/")


# get the end time
et = time.time()


# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')