from flask import Flask, request, render_template, session, redirect
from flask import render_template
import numpy as np
import pandas as pd
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
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import sys
from flask_sqlalchemy import SQLAlchemy
from jinja2 import Template


sys.path.append("/Users/juliafrench/Documents/MovieRecc/MovieGroupProj/movie_app/functions")

from functions import f

os.environ["JAVA_HOME"] = "/Users/juliafrench/Downloads/jdk-21.0.2.jdk/Contents/Home/"
os.environ["SPARK_HOME"] = "/Users/juliafrench/Documents/apache-spark/3.5.0/libexec"

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext
sc

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/recc")
def recc():
    df = f.movieTable()
    li = df.values.tolist()
    print(li)
    return render_template("recc.html", data=li)
    

@app.route("/Suggestions")
def get_data():
    return app.send_static_file("data.json")






