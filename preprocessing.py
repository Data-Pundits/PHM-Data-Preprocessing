"""
## INFO:
## This script reads all the raw data files from the 
## training folder and merges all the 3 sensors data 
## into a single data frame.

## This is the first script to be run to prepare the data for modeling.
## Next script to be run immediately after this is 'trimmer.py'.

## Alternatively, run the 'data-pipeline.sh' bash script in order to run all the 3 scripts in sequence.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window

import time

import datasource_config

#UDF to extract only the pressure values from the list.....
extract_values = F.udf(lambda row: list(map(float, row[1:])), T.ArrayType(T.FloatType(), containsNull=False))


# Function containing logic to read and consolidate data from multiple files one by one and union them into a single final dataframe.
def data_consolidator(base_path, file_type, individuals):
    df_final = None
    partition=Window.orderBy(F.monotonically_increasing_id())

    for index in individuals:
        path = base_path + file_type
        file_path = path + "_" + 'pdmp' + str(index) + ".csv"

        print("******READING FILE::::: ", 'pdmp' + str(index) + ".csv")
        df_pdmp = spark.read.text(file_path)
        df_pdmp = df_pdmp.select('value', F.split('value', ',').alias('value2'))
        df_pdmp = df_pdmp.withColumn('fault_class', df_pdmp['value2'][0])
        df_pdmp = df_pdmp.withColumn('pdmp', extract_values(df_pdmp['value2']))
        df_pdmp = df_pdmp.select('fault_class', 'pdmp')
        df_pdmp = df_pdmp.withColumn('index', F.row_number().over(partition))
        df_pdmp = df_pdmp.alias('df_pdmp')

        file_path = path + "_" + 'pin' + str(index) + ".csv"

        print("******READING FILE::::: ", 'pin' + str(index) + ".csv")
        df_pin = spark.read.text(file_path)
        df_pin = df_pin.select('value', F.split('value', ',').alias('value2'))
        df_pin = df_pin.withColumn('fault_class', df_pin['value2'][0])
        df_pin = df_pin.withColumn('pin', extract_values(df_pin['value2']))
        df_pin = df_pin.select('fault_class', 'pin')
        df_pin = df_pin.withColumn('index', F.row_number().over(partition))
        df_pin = df_pin.alias('df_pin')

        file_path = path + "_" + 'po' + str(index) + ".csv"

        print("******READING FILE::::: ", 'po' + str(index) + ".csv")
        df_po = spark.read.text(file_path)
        df_po = df_po.select('value', F.split('value', ',').alias('value2'))
        df_po = df_po.withColumn('fault_class', df_po['value2'][0])
        df_po = df_po.withColumn('po', extract_values(df_po['value2']))
        df_po = df_po.select('fault_class', 'po')
        df_po = df_po.withColumn('index', F.row_number().over(partition))
        df_po = df_po.alias('df_po')

        df_mv = df_pdmp.join(df_pin, df_pdmp['index'] == df_pin['index'], how='left').select('df_pdmp.index', 'df_pdmp.fault_class', 'df_pdmp.pdmp', 'df_pin.pin')
        df_mv = df_mv.alias('df_mv')
        df_mv = df_mv.join(df_po, df_mv['index'] == df_po['index'], how='left').select('df_mv.*', 'df_po.po')
        df_mv = df_mv.withColumn('individual', F.lit(index))

        if index == 1:
            df_final = df_mv
        else:
            df_final = df_final.unionByName(df_mv)
    return df_final


## This is the script's starting point:

# Create a SparkSession
spark = SparkSession \
    .builder \
    .appName("Data Stager") \
    .config("spark.driver.memory", '8g') \
    .getOrCreate()

# Set log level to ERROR to avoid unnesessary spark system logs.
spark.sparkContext.setLogLevel('ERROR')

# List of individual IDs that are present in the training dataset folder.
individuals = [1, 2, 4, 5, 6]

# Base relative path to the training dataset folder.
base_path = datasource_config.TRAINING_DATA_PATH

# Call the Consolidator function to read the files which start with the prefix 'data'
data_df = data_consolidator(base_path, 'data', individuals)
# Call the Consolidator function to read the files which start with the prefix 'ref'
ref_df = data_consolidator(base_path, 'ref', individuals)

# Union the two dataframes into a single one.
df_final = data_df.unionByName(ref_df)

# Get the start time
st = time.time()

# Partition key for adding the 'index' column.
partition=Window.orderBy(F.monotonically_increasing_id())

# Generate index using the row_number() spark function over the partition key.
df_final = df_final.withColumn('index', F.row_number().over(partition))

df_rdd = df_final.rdd
print(df_rdd.getNumPartitions())

# Repartition the final dataframe into 5 parts using 'individual' as the partitioning key as there are 5 individuals.
df_final = df_final.repartition(5, 'individual')

df_rdd = df_final.rdd
print(df_rdd.getNumPartitions())

# Write the final partitioned dataframe to the stage folder for next steps.
df_final.write.mode('overwrite').parquet("consolidated_stage/")

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')