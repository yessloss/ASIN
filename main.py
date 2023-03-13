import json
from collections import defaultdict, Counter
from datetime import datetime
from statistics import mean
import nltk
import ssl
import string
import pandas as pd
import gzip

from nltk.cluster import cosine_distance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")
nltk.download('wordnet')

import spacy

nlp = spacy.load("en_core_web_md")

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim import corpora
from gensim.models import LdaModel, KeyedVectors
from nltk import FreqDist, collocations, word_tokenize, NaiveBayesClassifier, accuracy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer





with open('/Users/szghhgh/Downloads/B07TDN9MKC.json', 'r') as file:
    data = json.load(file)
    reviews = []

sentiment_analyzer = SentimentIntensityAnalyzer()

sentiments = []
for review in data:
    sentiment = sentiment_analyzer.polarity_scores(review["body"])
    sentiment["name"] = review["name"]
    sentiment["title"] = review["title"]
    sentiment["stars"] = review["rating"]
    sentiment["body"] = review["body"]
    sentiment["date"] = review["date"]
    sentiment["verified"] = review["verified"]
    sentiments.append(sentiment)


positive_reviews = [review for review in sentiments if review["compound"] > 0.5]
negative_reviews = [review for review in sentiments if review["compound"] < -0.5]
neutral_reviews = [review for review in sentiments if review["compound"] >= -0.5 and review["compound"] <= 0.5]


average_result = [float(result["stars"]) for result in reviews if len(result["stars"]) >= 1]


good_phrases1 = [comment["body"] for comment in positive_reviews]
bad_phrases1 = [comment["body"] for comment in negative_reviews]
good_phrases = ["High quality", "Durable", "Affordable", "User-friendly",
            "Stylish design", "Excellent performance", "Versatile",
            "Comfortable to use", "Lightweight", "Great value for money",
            "High functionality", "Efficient", "Easy to clean", "Reliable",
            "Good customer support", "Improved features", "Saves time",
            "Innovative", "Energy efficient", "User-friendly interface",
            "Great battery life", "Convenient to carry", "Good grip",
            "Safe to use", "Multipurpose", "Good sound quality",
            "High-resolution display", "Fast processing speed", "Good connectivity options",
            "Affordable price"]
bad_phrases = ["Poor quality", "Unreliable", "Expensive", "Difficult to use",
            "Poor design", "Low performance", "Limited functionality", "Uncomfortable to use",
            "Heavy", "Poor value for money", "Low functionality", "Inefficient", "Hard to clean",
            "Unreliable", "Poor customer support", "Outdated features", "Wastes time",
            "Uninnovative", "Energy inefficient", "User-unfriendly interface",
            "Poor battery life", "Inconvenient to carry", "Poor grip",
            "Dangerous to use", "Limited purpose", "Poor sound quality",
            "Low-resolution display", "Slow processing speed",
            "Poor connectivity options", "Overpriced"]
good_result = []
bad_result = []
for comments in good_phrases1:
    for classification1 in good_phrases:
        doc1 = nlp(comments)
        doc2 = nlp(classification1)
        similarity = doc1.similarity(doc2)
        good_result.append([similarity, classification1, comments])
for comments in bad_phrases1:
    for classification2 in bad_phrases:
        doc1 = nlp(comments)
        doc2 = nlp(classification2)
        similarity = doc1.similarity(doc2)
        bad_result.append([similarity, classification2, comments])

reviews_by_month = {
    'positive': Counter(),
    'neutral': Counter(),
    'negative': Counter()
}

for review in data:
    date_str = review['date'].split()[-3:]
    month = date_str[0]

    sentiment_scores = sentiment_analyzer.polarity_scores(review['body'])
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        reviews_by_month['positive'][month] += 1
    elif compound_score <= -0.05:
        reviews_by_month['negative'][month] += 1
    else:
        reviews_by_month['neutral'][month] += 1


data_file = {
    "num_of_reviews": len(data),
    "num_of_positive_reviews": len(positive_reviews),
    "num_of_negative_reviews": len(negative_reviews),
    "num_of_neutral_reviews": len(neutral_reviews),
    "percent_of_positive_reviews": round(len(positive_reviews) / len(data) * 100),
    "percent_of_neutral_reviews": round(len(neutral_reviews) / len(data) * 100),
    "percent_of_negative_reviews": round(len(negative_reviews) / len(data) * 100),
}
with open("data.json", "w") as json_file:
    json.dump(data_file, json_file)
    json.dump(reviews_by_month, json_file)
#with open("report.txt", "w") as file:
    #file.write("Number of Reviews: {}\n".format(len(data)))
    #file.write("Number of Positive Reviews: {}\n".format(len(positive_reviews)))
    #file.write(f"Percent of Positive Reviews: {round(len(positive_reviews) / len(reviews) * 100)}%\n")
    #file.write("Number of Negative Reviews: {}\n".format(len(negative_reviews)))
    #file.write(f"Percent of Negative Reviews: {round(len(negative_reviews) / len(reviews) * 100)}%\n")
    #file.write("Number of Neutral Reviews: {}\n".format(len(neutral_reviews)))
    #file.write(f"Percent of Neutral Reviews: {round(len(neutral_reviews) / len(reviews) * 100)}%\n\n")
    #file.write("Top 3 most common positive comments:\n")
    #for mood in most_common_moods:
    #file.write("- '{}'\n".format(sentiment2))

    #file.write("Top positive sentiment analysis")
    #positive_result_dict = {}
    #for item in good_result:
        #result2, value, comment = item
        #if comment in positive_result_dict:
            #if result2 > positive_result_dict[comment][0]:
                #positive_result_dict[comment] = [result2, value]
        #else:
            #positive_result_dict[comment] = [result2, value]
    #for comment, (result2, value) in positive_result_dict.items():
        #file.write(f"{value}\n")
    #file.write("Top negative sentiment analysis")
    #negative_result_dict = {}
    #for item in bad_result:
        #result2, value, comment = item
        #if comment in negative_result_dict:
            #if result2 > negative_result_dict[comment][0]:
                #negative_result_dict[comment] = [result2, value]
        #else:
            #negative_result_dict[comment] = [result2, value]
    #for comment, (result2, value) in negative_result_dict.items():
        #file.write(f"{value}\n")
    #for comment in most_common_negative_comments:
        #file.write("- '{}'\n".format(comment))
    #for title in most_common_positive_titles:
        #file.write("- '{}'".format(title))
        #file.write(f" {positive_titles.count(title)} times\n")
    #file.write("\nNegative reviews analysis:\n")
    #for title in most_common_negative_titles:
        #file.write("- '{}'".format(title))
        #file.write(f" {negative_titles.count(title)} times\n")
    #if len(positive_reviews) > len(negative_reviews) and len(positive_reviews) > len(neutral_reviews):
        #file.write("This company good, have some neutral "
                   #"and negative comments, but positive comments are more")
    #if len(neutral_reviews) > len(negative_reviews) and len(neutral_reviews) > len(positive_reviews):
        #file.write("This company aren't bad, but aren't good too, "
                   #"you can can see some comments about your company and fix them")
    #if len(negative_reviews) > len(neutral_reviews) and len(negative_reviews) > len(positive_reviews):
        #file.write("This company have some problems you must fix them, read negative comments")



