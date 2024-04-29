from flask import Flask, request, render_template, session, redirect, url_for
from flask import render_template
import numpy as np
import pandas as pd

import os
from pyspark.sql import SparkSession
import sys
from functions import f



sys.path.append("/Users/juliafrench/Documents/MovieRecc/MovieGroupProj/movie_app/functions")

os.environ["JAVA_HOME"] = "/Users/juliafrench/Downloads/jdk-21.0.2.jdk/Contents/Home/"
os.environ["SPARK_HOME"] = "/Users/juliafrench/Documents/apache-spark/3.5.0/libexec"

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext
sc

app = Flask(__name__)
app.secret_key = "secretkey"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/recc")
def recc():
    df = f.movieTable()
    li = df.values.tolist()
    print(li)
    return render_template("recc.html", data=li)
    
@app.route('/process', methods=['GET','POST']) 
def process(): 

    if request.method == "POST":
        # get selected ids from recc.html
        data = request.form.get('data') 
        # use ids to get recommended movies: f.py
        reccomendations = f.selToList(data)
        recs = reccomendations.values.tolist()
        session["recMovies"] = recs
        return redirect(url_for('suggestions'))
    return render_template('recc.html')


@app.route('/suggestions')
def suggestions():
    movies = session['recMovies']  
  
    # make table with recommended movies
    return render_template("suggestions.html", data=movies)
    
    






