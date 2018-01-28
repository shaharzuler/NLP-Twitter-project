
"""
@author: Shahar Zuler
"""
import torch
import nltk
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
nltk.download('punkt')

def cosine(u, v):
    return (np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

infersent = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)
glove_path='glove.840B.300d.txt'
infersent.set_glove_path(glove_path)#########

df2=pd.DataFrame.from_csv('ObTr1.csv', encoding='cp1252')
sentences = df2["data_tweets"].tolist()

infersent.build_vocab(sentences, tokenize=True)
infersent.update_vocab(sentences)
embeddings = infersent.encode(sentences, tokenize=True)
#infersent.visualize(sentences[100], tokenize=True)

sim = np.zeros((len(sentences), len(sentences)))
for i in range(len(sentences)):
    for j in range(len(sentences)):
        sim[i, j] = cosine( embeddings[i, :], embeddings[j, :] )

km=KMeans()
y_km_new= km.fit(embeddings)
y=k_means(X=embeddings, n_clusters=2)
y_pred = y[1]
y2=np.zeros((400))
y2[:200] = 1
y3=dbscan(X=embeddings)

acc=max(sum(y_pred==y2) / df2.shape[0], 1-sum(y_pred==y2) / df2.shape[0])

print('DONE!')
print('Accuracy is: ' + str (acc))
