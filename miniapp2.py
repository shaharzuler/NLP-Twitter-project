# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 19:10:32 2018

@author: User
"""
import pandas as pd     
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import random
from sklearn.externals import joblib
import pickle





def review_to_words( raw_review ):
    '''function for preproceccing the sentences'''
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    # 4.  convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]  
    #6. stemmer
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in meaningful_words]
    #7. lemmitizer
    wordnet_lemmatizer = WordNetLemmatizer()
    lammatized_words = [wordnet_lemmatizer.lemmatize(word) for word in  stemmed_words  ]
    # 8. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   


model = joblib.load('SVMmodel.pkl') 
#vectorizer = joblib.load('vectorizer.pkl') 

train=pd.DataFrame.from_csv('ObTr_app.csv', encoding='cp1252')  

num=random.randint(0,len(train)-1)
print (num)
X=train['data_tweets'][num]
#X='All across America people chose to get involved, get engaged and stand up. Each of us can make a difference, and all of us ought to try. So go keep changing the world in 2018.'
print(X)

clean_review = review_to_words(X)
data_feature = vectorizer.transform([clean_review])
data_feature = data_feature.toarray()


pred =model.predict(data_feature)[0]
truth = train['data_labels'][num]
print(pred == truth)





    
    
    
    
    
    
    
    
    
    
    