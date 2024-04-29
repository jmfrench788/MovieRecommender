
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

def getReccomendations(ids, numRec):
    ids = ids
    cosine_sim = transformClean()
    num_recommendations = numRec

    df = movieTable()
    movieDF = df
    movieDF['combined_features'] =  movieDF['Genre'] + ' ' + movieDF['Sub_Genre'] + ' ' + movieDF['Director'] + ' ' + movieDF['Year'].astype(str)

    recommendations = []
    recs = []
    for movie_id in ids:
        print(movie_id)
        # Get the index of the movie that matches the ID
        idx = movieDF.index[movieDF['movieID'] == movie_id]
        if len(idx) > 0:
            idx = idx[0]
            # Get the pairwise similarity scores
            sim_scores = list(enumerate(cosine_sim[idx]))

            # Sort the movies based on similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the scores of the most similar movies
            sim_scores = sim_scores[1:num_recommendations+1]

            # Get the movie indices
            movie_indices = [i[0] for i in sim_scores]

            # Add recommendations for this movie to the list
            recommendations.append(movieDF['movieID'].iloc[movie_indices].tolist())
        else:
            recommendations.append([])  
          
            # Movie not found in the dataset

        recs = recommendations

        flat_list = []
        for row in recs:
            flat_list.extend(row)
    
        res = []
        [res.append(x) for x in flat_list if x not in res]

        rows = df[df['movieID'].isin(res)]

    return rows





def selToList(li):
    li = li
    # make list of selected ids
    newSt = li.replace("\"", "")
    newSt = newSt.replace("[", "")
    newSt = newSt.replace("]", "")
    splitSt = newSt.split(",")

    ids = [int(i) for i in splitSt]


    lenID = len(ids)
    if lenID == 1:
        numRec = 10
    elif lenID == 2:
        numRec = 5
    elif lenID == 3:
        numRec = 4
    elif lenID == 4:
        numRec = 3
    elif lenID == 5:
        numRec = 2
    elif lenID == 6:
        numRec = 2
    else:
        numRec = 1
    
    recommendations = getReccomendations(ids, numRec)
    return recommendations



