# -*- coding: utf-8 -*-
"""
Created on Sat May  1 14:11:53 2021

@author: FaSa79_
"""

import nltk
import string
import os
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import scipy.sparse as sp
import numpy as np

target = ['game', 'declin', 'trade', 'win', 'terror']

path = 'D:/OneDrive - International Islamic University Malaysia/Studies/3rd Year 2nd Sem/CSC4309 Natural Language Processing/Assignment2/article'

token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        article = open(file_path, 'r')
        text = article.read()
        text = text.lower()
        text = text.translate(string.punctuation)
        text = text.replace('+', ' ')
        text = text.replace(',', '')
        token_dict[file] = text

tfidf = TfidfVectorizer(tokenizer = tokenize, stop_words=stopwords.words('english'))
#tfidf = TfidfVectorizer(use_idf = True)
tfs = tfidf.fit_transform(token_dict.values())
features = tfidf.get_feature_names()
n = len(features)

inverse_idf = sp.diags(1 / tfidf.idf_, offsets = 0, shape = (n, n), format = 'csr', dtype = np.float64).toarray()
df_tf = pd.DataFrame(tfs*inverse_idf, columns = features, index = files)
df_tf = df_tf.transpose()

df_idf = pd.DataFrame(tfidf.idf_, index = features, columns=["IDF"])

df_tfidf = pd.DataFrame(tfs.T.todense(), index = features, columns = files)

print("\nTF Score")
print(df_tf.loc[target])
print("\nIDF Score")
print(df_idf.loc[target])
print("\nTF-IDF Score")
print(df_tfidf.loc[target])
"""
def k_means(tfs):
    true_k = 2
    model = KMeans(n_clusters = true_k, init='k-means++', max_iter=50, n_init=1)
    model.fit(tfs)
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    
    for i in range(true_k):
        print("Cluster %d: " % i)
        for ind in order_centroids[i, :10]:
            print(" %s" % terms[ind])

k_means(tfs)
"""

