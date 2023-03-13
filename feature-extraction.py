from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window
from scipy.stats import skew
from scipy.stats import kurtosis

import time
import numpy as np


WINDOW_SIZE_150 = 150
ARRAY_SIZE = 571

#UDF to extract only the pressure values from the list.....
average_for_window = F.udf(lambda row, from_index, to_index: float(np.mean(row[from_index:to_index])), T.FloatType())
variance_for_window = F.udf(lambda row, from_index, to_index: float(np.var(row[from_index:to_index])), T.FloatType())
skew_for_window = F.udf(lambda row, from_index, to_index: float(skew(row[from_index:to_index])), T.FloatType())
kurtosis_for_window = F.udf(lambda row, from_index, to_index: float(kurtosis(row[from_index:to_index])), T.FloatType())
# print_array = F.udf(lambda row: print(row, "\n", row[571-LAST_50_WINDOW_SIZE:]), T.FloatType())


def calculate_metrics(phm_df, sensor_name):
    phm_df = phm_df.withColumn(sensor_name + '_avg_first_150', average_for_window(phm_df[sensor_name], F.lit(0), F.lit(WINDOW_SIZE_150)))
    phm_df = phm_df.withColumn(sensor_name + '_avg_last_150', average_for_window(phm_df[sensor_name], F.lit(array_size-WINDOW_SIZE_150), F.lit(array_size)))
    phm_df = phm_df.withColumn(sensor_name + '_avg_mid_300', average_for_window(phm_df[sensor_name], F.lit(200), F.lit(500)))
    phm_df = phm_df.withColumn(sensor_name + '_avg_diff', phm_df[sensor_name + '_avg_first_150'] - phm_df[sensor_name + '_avg_last_150'])

    phm_df = phm_df.withColumn(sensor_name + '_size', F.size(sensor_name))
    phm_df = phm_df.withColumn(sensor_name + '_size', phm_df[sensor_name + '_size'].cast(T.FloatType()))
    phm_df = phm_df.withColumn(sensor_name + '_total_avg', F.expr('aggregate(' + sensor_name + ', 0F, (total, x) -> total + x, total -> total / ' + sensor_name + '_size)'))

    phm_df = phm_df.withColumn(sensor_name + '_variance', variance_for_window(phm_df[sensor_name], F.lit(0), F.lit(ARRAY_SIZE)))
    phm_df = phm_df.withColumn(sensor_name + '_skew', skew_for_window(phm_df[sensor_name], F.lit(0), F.lit(ARRAY_SIZE)))
    phm_df = phm_df.withColumn(sensor_name + '_kurtosis', kurtosis_for_window(phm_df[sensor_name], F.lit(0), F.lit(ARRAY_SIZE)))

    return phm_df

# Create a SparkSession
spark = SparkSession \
    .builder \
    .appName("Data Stager") \
    .config("spark.driver.memory", '8g') \
    .getOrCreate()

spark.sparkContext.setLogLevel('ERROR')

# get the start time
st = time.time()

phm_df = spark.read.parquet("../output/")

array_size = phm_df.select(F.size(F.col('pdmp'))).collect()[0][0]

phm_df = calculate_metrics(phm_df, 'pdmp')
phm_df = calculate_metrics(phm_df, 'pin')
phm_df = calculate_metrics(phm_df, 'po')

phm_df = phm_df.withColumn('fault_class', phm_df['fault_class'].cast(T.IntegerType()))

phm_df.show()
phm_df.printSchema()


phm_df = phm_df.repartition(5, 'individual')
phm_df.write.mode('overwrite').parquet("../feature_extracts/")


# get the end time
et = time.time()


# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')