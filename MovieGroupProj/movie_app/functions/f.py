
import pandas as pd

import os
from json import loads, dumps
from localStoragePy import localStoragePy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



os.environ["JAVA_HOME"] = "/Users/juliafrench/Downloads/jdk-21.0.2.jdk/Contents/Home/"
os.environ["SPARK_HOME"] = "/Users/juliafrench/Documents/apache-spark/3.5.0/libexec"

localStorage = localStoragePy('movie_app', 'text')

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext
sc

def movieTable():
    sparkDF = spark.read.csv('/Users/juliafrench/Documents/MovieRecc/IMDb_Clean_CSV.csv', header=True, inferSchema=True)
    df = sparkDF.toPandas()
    
    return df


# Concatenate the features into one string (ensure these columns exist in your DataFrame)
def transformClean():

    dataframe = movieTable()

    dataframe['combined_features'] = dataframe['Genre'] + ' ' + dataframe['Sub_Genre'] + ' ' + dataframe['Director'] + ' ' + dataframe['Year'].astype(str)

    # Convert textual data to numerical data using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['combined_features'])

    # Compute the cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim


def selToList(li):
    # make list of selected ids
    newSt = li.replace("\"", "")
    newSt = newSt.replace("[", "")
    newSt = newSt.replace("]", "")
    splitSt = newSt.split(",")

    ids = [int(i) for i in splitSt]
    
    df = movieTable()
    # get first id 
    firstID = ids[0]
    # Get rows that match based on id
        # rows = df[df['movieID'].isin(ids)]

    movieDF = df

    movieDF['combined_features'] =  movieDF['Genre'] + ' ' + movieDF['Sub_Genre'] + ' ' + movieDF['Director'] + ' ' + movieDF['Year'].astype(str)

    cosine_sim = transformClean()
    # Get the index of the movie that matches the title
    idx = movieDF[movieDF['movieID'] == firstID].index[0] 
    print(idx)

    cosine_sim = transformClean()

    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the most similar movies
    num_recommendations=10
    sim_scores = sim_scores[1:num_recommendations+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    
    reccs = movieDF['movieID'].iloc[movie_indices].tolist()

    # get rows which match the ids of recommended movies
    movies = df[df['movieID'].isin(reccs)]
    # Return the top 10 most similar movies
    return movies



  







    
    


