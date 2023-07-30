import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error,confusion_matrix,
precision_score, recall_score, auc,roc_curve
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import math
from collections import Counter
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import
classification_report,confusion_matrix,accuracy_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
for filename in filenames:
print(os.path.join(dirname, filename))
import socket
from datetime import datetime
import time
from bs4 import BeautifulSoup
import urllib
import bs4
dataset=pd.read_csv("malicious_url.csv")
dataset.head()
def D_C_ratio(url):
count_numbers= sum(item.isdigit() for item in url)
Char_len=len(url)
r = count_numbers/Char_len
return(r)
dataset['Digits_to_char_ratio'] = dataset['url'].apply(D_C_ratio)
def L_C_ratio(url):
Lowercase= sum(item.islower() for item in url)
length=len(url)
ratio = Lowercase/length
return(ratio)
dataset['Lcase_to_char_ratio'] = dataset['url'].apply(L_C_ratio)
def No_of_param(url):
param =url.split('&')
return len(param)-1
dataset['No_of_parameters'] = dataset['url'].apply(No_of_param)
import math
def url_entropy(url):
s= url.strip()
prob= [float(s.count(c))/len(s)for c in dict.fromkeys(list(s))]
entropy= sum([(p * math.log(p)/ math.log(2.0))for p in prob])
return entropy
dataset['Url_entropy'] = dataset['url'].apply(url_entropy)
def has_http(url):
return 'http:' in url
dataset['Has_http'] = dataset['url'].apply(has_http)
from urllib.parse import urlparse
def path_len(url):
return len(urlparse(url).path)
dataset['Path_length'] = dataset['url'].apply(path_len)
from urllib.parse import urlparse
def host_len(url):
return len(urlparse(url).netloc)
dataset['Host_length'] = dataset['url'].apply(host_len)
def no_of_frags(url):
frags=url.split('#')
return len(frags)-1
dataset['No_of_fragments'] = dataset['url'].apply(no_of_frags)
def no_of_subdomains(url):
sd=url.split('http')[-1].split('//')[-1].split('/')
return len(sd)-1
dataset['No_of_Subdomains'] = dataset['url'].apply(no_of_subdomains)
dataset.head()
import tensorflow as tf
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
enc= preprocessing.LabelEncoder()
dataset["type"]=enc.fit_transform(dataset["type"])
dataset.loc[dataset["Has_http"]==False,"Hs_http"]=0
dataset.loc[dataset["Has_http"]==True,"Hs_http"]=1
dataset.head()
dataset.drop(['Has_http'],axis=1,inplace=True)
dataset.head()
X = dataset.iloc[:,3:-1].values
X
Y=dataset.iloc[:,2]
Y
X_train,X_test,Y_train,Y_test =
train_test_split(X,Y,test_size=0.33,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
ann_model = tf.keras.models.Sequential()
ann_model.add(tf.keras.layers.Dense(units=10,activation="relu"))
ann_model.add(tf.keras.layers.Dense(units=20,activation="relu"))
ann_model.add(tf.keras.layers.Dense(units=10,activation="relu"))
ann_model.add(tf.keras.layers.Dense(units=4,activation="relu"))
ann_model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCr
ossentropy(from_logits=True),metrics=['accuracy'])
ann_model.fit(X_train,Y_train,batch_size=200,epochs =
10,validation_data=(X_test, Y_test))
ann_model.save("ANN_model")
