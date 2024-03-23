from pyspark.mllib.classification import LogisticRegressionModel,LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.clustering import *
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, mean, split
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline 
from pyspark.ml.classification import LogisticRegression
import pyspark.sql.functions as F
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import *
from pyspark.sql import Row
import os
from json import loads, dumps


os.environ["JAVA_HOME"] = "/Users/juliafrench/Downloads/jdk-21.0.2.jdk/Contents/Home/"
os.environ["SPARK_HOME"] = "/Users/juliafrench/Documents/apache-spark/3.5.0/libexec"

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext
sc

def movieTable():
    sparkDF = spark.read.csv('/Users/juliafrench/Documents/MovieRecc/IMDb_Clean_CSV.csv', header=True, inferSchema=True)
    df = sparkDF.toPandas()
    
    return df






    
    


