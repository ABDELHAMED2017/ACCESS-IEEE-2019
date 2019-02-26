import numpy as np
import os
from pyspark.sql import SparkSession
import calendar
import sys
import gc
import datetime

spark = SparkSession.builder.appName("Operations").getOrCreate()

assert len(sys.argv) >= 2, "Falta directorio de datos"
project_dir = sys.argv[1]
if project_dir.endswith("/"):
    project_dir = project_dir[:-1]

assert os.path.exists(project_dir)

delta_next_week = datetime.timedelta(days=7)
delta_end_range = datetime.timedelta(days=6, hours=23, minutes=59, seconds=59)

from_date = datetime.datetime(year=2018, month=2, day=1)
from_date = from_date - delta_next_week

for num_week in range(15):
    from_date = from_date + delta_next_week
    to_date = from_date + delta_end_range

    from_date_s = "{:02d}-{:02d}-{:02d}".format(from_date.year, from_date.month, from_date.day)
    to_date_s = "{:02d}-{:02d}-{:02d}".format(to_date.year, to_date.month, to_date.day)
    build_dir = '{}.from-{}-to-{}/'.format(project_dir, from_date_s, to_date_s)

    command = "python3 ~/data-tools/slice-date-range/slice-date-range.py '{}' {} {}".format(project_dir, from_date_s,
                                                                                            to_date_s)
    print("Genero el siguiente comando", command)
    os.system(command)

    print("Descomprimiendo data1")
    os.chdir(build_dir)
    if not os.path.exists("data.csv"):
        command = "gunzip -k data.csv.gz"
        os.system(command)

    df1 = spark.read.csv(build_dir + "data.csv", inferSchema=True, header=True)
    df1_co = df1.filter(df1["sensor"] == "co").filter(df1["node_id"] == "001e0610f732").select([df1["node_id"], df1["value_hrf"].cast("float")])
    df1_co = df1_co.na.drop(how="any", subset=["value_hrf"])
    df1_co.show()
    df1_co.toPandas().to_csv("/home/ruben/data_filtered_{:03d}.csv".format(num_week))

    os.chdir("../")