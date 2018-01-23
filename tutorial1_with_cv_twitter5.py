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
from sklearn.decomposition import PCA
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

train=pd.DataFrame.from_csv('ObTr1.csv', encoding='cp1252')
  
train2 = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
X=np.array(train['data_tweets'])
y=np.array(train['data_labels'])
models = []
ACCs = []
models2 = []
ACCs2 = []
models3 = []
ACCs3 = []
best_acc = 0
best_acc2 = 0
best_acc3 = 0

for j in range(10):
    X_train, X_cv_temp, y_train, y_cv_temp = train_test_split(X, y, test_size=0.125, random_state=42)
    X_cv, X_test, y_cv, y_test = train_test_split(X_cv_temp, y_cv_temp, test_size=0.5, random_state=42)

    
    
    # Get the number of reviews based on the dataframe column size
    num_reviews = X_train.size
    
    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []
    
    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list 
    for i in range( 0, num_reviews ):
        # If the index is evenly divisible by 1000, print a message
        if( (i+1)%1000 == 0 ):
            print ("Review %d of %d\n" % ( i+1, num_reviews ))   
        # Call our function for each one, and add the result to the list of
        # clean reviews
        clean_train_reviews.append( review_to_words( X_train[i] ) )
        
    print ("Creating the bag of words...\n")
    
    
    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 20000) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    
    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()
    #print (train_data_features.shape)
    
    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    
    
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)
    
    # For each, print the vocabulary word and the number of times it 
    # appears in the training set
    #for tag, count in zip(vocab, dist):
        #if count>30:
            #print (count, tag)
        
    print ("Training the random forest...")
    
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100) 
    clf = MultinomialNB()
    clf_sgd = SGDClassifier(loss='perceptron', penalty='l2', alpha=6e-4, random_state=42) #max_iter=5, ,  tol=None
    #loss 'hinge' on SGD used to give 96 acc on cv but only 92 on test
    
    # Fit the forest to the training set, using the bag of words as features and the sentiment labels as the response variable
    forest = forest.fit( train_data_features, y_train )
    clf = clf.fit(train_data_features, y_train)
    clf_sgd = clf_sgd.fit(train_data_features, y_train)
    models.append(forest)
    models2.append(clf)
    models3.append(clf_sgd)
    
    train_result = forest.predict(train_data_features)
    train_result2 = clf.predict(train_data_features)
    train_result3 = clf_sgd.predict(train_data_features)

    
    acc_train=np.sum((train_result==y_train))/len(X_train)
    acc_train2=np.sum((train_result2==y_train))/len(X_train)
    acc_train3=np.sum((train_result3==y_train))/len(X_train)

    
    print('forest train result: ' + str(acc_train))
    print('naive bias train result: ' + str(acc_train2))
    print('SGD train result: ' + str(acc_train2))



    
    # Create an empty list and append the clean reviews one by one
    num_reviews = len(X_cv)
    clean_cv_reviews = [] 
    
    print ("Cleaning and parsing the cv set movie reviews...\n")
    for i in range(0,num_reviews):
        if( (i+1) % 1000 == 0 ):
            print ("Review %d of %d\n" % (i+1, num_reviews))
        clean_review = review_to_words( X_cv[i] )
        clean_cv_reviews.append( clean_review )
    
    # Get a bag of words for the cv set, and convert to a numpy array
    cv_data_features = vectorizer.transform(clean_cv_reviews)
    cv_data_features = cv_data_features.toarray()
    
    # Use the random forest to make sentiment label predictions
    result = forest.predict(cv_data_features)
    #use naive bias
    result2 = clf.predict(cv_data_features)
    #use SGD
    result3 = clf_sgd.predict(cv_data_features)
    
    acc=np.sum((result==y_cv))/len(X_cv)
    acc2=np.sum((result2==y_cv))/len(X_cv)
    acc3=np.sum((result3==y_cv))/len(X_cv)

        
    print("ACC of forest model - CV "+ str(j) + ': ' + str(acc))
    print("ACC of naive bias model - CV "+ str(j) + ': ' + str(acc2))
    print("ACC of SGD model - CV "+ str(j) + ': ' + str(acc3))
    
    ACCs.append(acc)
    ACCs2.append(acc2)
    ACCs3.append(acc3)
    if acc> best_acc:
        best_acc=acc
        best_model = forest
        X_test_best = X_test
        y_test_best = y_test
    if acc2> best_acc2:
        best_acc2=acc2
        best_model2 = clf
        X_test_best2 = X_test
        y_test_best2 = y_test
    if acc3> best_acc3:
        best_acc3=acc3
        best_model3 = clf_sgd
        X_test_best3 = X_test
        y_test_best3 = y_test

#saving x,y_test_best3 to csv for future app

df_app=pd.DataFrame()
df_app[0]=X_test_best3
df_app[1]=y_test_best3
df_app.columns=['data_tweets','data_labels']

##to do - spemming

df_app.to_csv('ObTr_app.csv')


print('DONE!')
print (np.average(np.array(ACCs)))
print (np.average(np.array(ACCs2)))
print (np.average(np.array(ACCs3)))

#print(list(zip(models,ACCs)))

print('best forest model is:')
print( best_model)
print('best acc forest is:' + str(best_acc))

print('best naive bias model is:')
print( best_model2)
print('best acc naive bias is:' + str(best_acc2))


print('best SGD model is:')
print( best_model3)
print('best acc SGD is:' + str(best_acc3))

#preprocess test data:
    
num_reviews = len(X_test_best)
clean_test_reviews = [] 

print ("Cleaning and parsing the *test* set movie reviews...\n")
for i in range(num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( X_test_best[i] )
    clean_test_reviews.append( clean_review )


# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
test_result = best_model.predict(test_data_features)
test_result2 = best_model2.predict(test_data_features)
test_result3 = best_model3.predict(test_data_features)


test_acc=np.sum((test_result==y_test_best))/len(X_test_best)
test_acc2=np.sum((test_result2==y_test_best2))/len(X_test_best2)
test_acc3=np.sum((test_result3==y_test_best3))/len(X_test_best3)


print("ACC of test model forest : " + str(test_acc))
print("ACC of test model naive bias : " + str(test_acc2))
print("ACC of test model SGD : " + str(test_acc3))

#visualization of training set:
pca = PCA(n_components=10)
pca.fit(train_data_features.T)
a=pca.components_
pca_var = pca.explained_variance_ 
print(pca_var)
fig = plt.figure()
ax =fig.add_subplot(111, projection='3d')
for i in range(len(y_train)):
   if y_train[i] ==1:
       c = 'b'
   elif y_train[i] ==2:
       c = 'g'
   else:
       c = 'r'   
   ax.scatter(a[0][i],a[1][i], a[2][i],color = c)
   
plt.show()

#saving my model:

s1 = pickle.dumps(best_model3)
s2 = pickle.dumps(vectorizer)

joblib.dump(best_model3, 'SVMmodel.pkl')
joblib.dump(best_model3, 'vectorizer.pkl')

#later use:
#from sklearn.externals import joblib
#import pickle
#model = joblib.load('SVMmodel.pkl') 
#vectorizer = joblib.load('vectorizer.pkl') 





    
    
    
    
    
    
    
    
    
    
    