"""
## INFO:
## This script generates engineered features (simple and advanced statistical measures) from the raw dataset.

## It requires the 'trimmer.py' to be run prior to this and some data should be present in the 'output/' folder.
## This is the last job in the data-engineering pipeline.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window
from scipy.stats import skew, kurtosis, ttest_1samp, ttest_ind, kruskal

import time
import numpy as np


WINDOW_SIZE_150 = 150
ARRAY_SIZE = 571

#UDF to extract only the pressure values from the list.....
average_for_window = F.udf(lambda row, from_index, to_index: float(np.mean(row[from_index:to_index])), T.FloatType())
variance_for_window = F.udf(lambda row, from_index, to_index: float(np.var(row[from_index:to_index])), T.FloatType())
skew_for_window = F.udf(lambda row, from_index, to_index: float(skew(row[from_index:to_index])), T.FloatType())
kurtosis_for_window = F.udf(lambda row, from_index, to_index: float(kurtosis(row[from_index:to_index])), T.FloatType())

#Two-Way T Test
twt_test = F.udf(lambda row, mean: float(ttest_1samp(row, mean).statistic), T.FloatType())

# Kruskal Wallis  U Test
kwu_test = F.udf(lambda sensor1, sensor2: float(ttest_ind(sensor1, sensor2).statistic), T.FloatType())

# Kruskal Wallis H Test
kwh_test = F.udf(lambda sensor1, sensor2, sensor3: float(kruskal(sensor1, sensor2, sensor3).statistic), T.FloatType())


def calculate_metrics(phm_df, sensor_name):
    # Basic statistical measures:
    phm_df = phm_df.withColumn(sensor_name + '_avg_first_150', average_for_window(phm_df[sensor_name], F.lit(0), F.lit(WINDOW_SIZE_150)))
    phm_df = phm_df.withColumn(sensor_name + '_avg_last_150', average_for_window(phm_df[sensor_name], F.lit(array_size-WINDOW_SIZE_150), F.lit(array_size)))
    phm_df = phm_df.withColumn(sensor_name + '_avg_mid_300', average_for_window(phm_df[sensor_name], F.lit(200), F.lit(500)))
    phm_df = phm_df.withColumn(sensor_name + '_avg_diff', phm_df[sensor_name + '_avg_first_150'] - phm_df[sensor_name + '_avg_last_150'])

    phm_df = phm_df.withColumn(sensor_name + '_size', F.size(sensor_name))
    phm_df = phm_df.withColumn(sensor_name + '_size', phm_df[sensor_name + '_size'].cast(T.FloatType()))
    phm_df = phm_df.withColumn(sensor_name + '_total_avg', F.expr('aggregate(' + sensor_name + ', 0F, (total, x) -> total + x, total -> total / ' + sensor_name + '_size)'))

    # Variance, Skewness and Kurtosis measures:
    phm_df = phm_df.withColumn(sensor_name + '_variance', variance_for_window(phm_df[sensor_name], F.lit(0), F.lit(ARRAY_SIZE)))
    phm_df = phm_df.withColumn(sensor_name + '_skew', skew_for_window(phm_df[sensor_name], F.lit(0), F.lit(ARRAY_SIZE)))
    phm_df = phm_df.withColumn(sensor_name + '_kurtosis', kurtosis_for_window(phm_df[sensor_name], F.lit(0), F.lit(ARRAY_SIZE)))
    
    # Two-way T-Test calculation:
    phm_df = phm_df.withColumn(sensor_name + '_twtt', twt_test(phm_df[sensor_name], phm_df[sensor_name + '_total_avg']))

    return phm_df


def calculate_kruskal_test_metrics(phm_df):
    # Calculating Kruskal-Wallis U Test for 3 different combinations of the sensors:
    phm_df = phm_df.withColumn('pdmp_pin' + '_kwut', kwu_test(phm_df['pdmp'], phm_df['pin']))
    phm_df = phm_df.withColumn('pdmp_po' + '_kwut', kwu_test(phm_df['pdmp'], phm_df['po']))
    phm_df = phm_df.withColumn('pin_po' + '_kwut', kwu_test(phm_df['pin'], phm_df['po']))

    # Calculating Kruskal-Wallis H Test for all the 3 sensors:
    phm_df = phm_df.withColumn('pdmp_pin_po' + '_kwht', kwh_test(phm_df['pdmp'], phm_df['pin'], phm_df['po']))

    return phm_df


# Create a SparkSession
spark = SparkSession \
    .builder \
    .appName("Data Stager") \
    .config("spark.driver.memory", '8g') \
    .config("spark.jars", "postgresql_jars/postgresql-42.2.18.jar") \
    .getOrCreate()

spark.sparkContext.setLogLevel('ERROR')

# get the start time
st = time.time()

phm_df = spark.read.parquet("consolidated_trimmed_stage/")

array_size = phm_df.select(F.size(F.col('pdmp'))).collect()[0][0]

phm_df = calculate_metrics(phm_df, 'pdmp')
phm_df = calculate_metrics(phm_df, 'pin')
phm_df = calculate_metrics(phm_df, 'po')

phm_df = calculate_kruskal_test_metrics(phm_df)

#Writing to Parquet files:
phm_df = phm_df.withColumn('fault_class', phm_df['fault_class'].cast(T.IntegerType()))

#Writing to DB:
phm_df = phm_df.repartition(5, 'individual')
phm_df.write.mode('overwrite').parquet("feature_extracts/")
phm_df.write.format('jdbc').option("url", "jdbc:postgresql://localhost:5432/postgres") \
    .option("driver", "org.postgresql.Driver").option("dbtable", "features_extracted") \
    .option("user", "postgres").option("password", "postgres").save()



# get the end time
et = time.time()


# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')