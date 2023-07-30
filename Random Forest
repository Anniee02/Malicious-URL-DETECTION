import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error,confusion_matrix,
precision_score,recall_score, auc,roc_curve
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
import pickle
dataset=pd.read_csv("C:\\Users\\hp\\Downloads\\malicious_url.csv")
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
 return 'http' in url
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
import re
def suspicious_words(url):
 match
=re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|eba
yisapi|webscr',url)
 if match:
 return 1
 else:
 return 0
dataset['sus_url'] = dataset['url'].apply(lambda i: suspicious_words(i))
def digit_count(url):
 digits = 0
 for i in url:
 if i.isnumeric():
 digits = digits + 1
 return digits
dataset['count-digits']= dataset['url'].apply(lambda i: digit_count(i))
def letter_count(url):
 letters = 0
 for i in url:
 if i.isalpha():
 letters = letters + 1
 return letters
dataset['count-letters']= dataset['url'].apply(lambda i: letter_count(i))
dataset.head()
dataset.to_csv('url_features.csv')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
ds = pd.read_csv('url_features.csv')
print('The shape of our dataset is:', ds.shape)
ds.describe()
ds.loc[ds["type"]== 'benign', "type1"]= int(0)
ds.loc[ds["type"]!= 'benign', "type1"]= int(1)
ds=ds.astype({"type1": "int32"})
header=["Number","No","url","type","f1","f2","f3","f4","f5","f6","f7","f8","f9"
,"f10","f11","f12","f13","type1"]
ds.columns=header
ds.drop(["Number","No","url", "type"],axis=1,inplace=True)
ds.head()
enc= preprocessing.LabelEncoder()
ds["f6"]=enc.fit_transform(ds["f6"])
ds["type1"]=enc.fit_transform(ds["type1"])
ds.head()
X=ds[["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12"]]
Y=ds["type1"]
#Splitting the Dataset into Training and Test Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
test_size=0.3,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)
#Import the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
RFReg = RandomForestRegressor(n_estimators = 100, random_state = 0)
#Fitting
RFReg.fit(X_train, Y_train)
Y_test=Y_test.to_numpy()
type(Y_test)
Y_predict_rfr = RFReg.predict((X_test))
Y_predict_rfr=Y_predict_rfr.astype(np.int32)
#Model Evaluation for Random Forest Regression
from sklearn import metrics
accuracy = metrics.accuracy_score(Y_test,Y_predict_rfr)
print('Model accuracy: ', accuracy*100)
import pickle
with open ('model_pkl',"wb")as file:
 pickle.dump(RFReg,file)
def predict_mal(url):
 f=[]
 a= D_C_ratio(url)
 b= L_C_ratio(url)
 c= No_of_param(url)
 d= url_entropy(url)
 e= has_http(url)
 p= path_len(url)
 g= host_len(url)
 h= no_of_frags(url)
 i= no_of_subdomains(url)
 j= suspicious_words(url)
 k= digit_count(url)
 m=len(url)

 f.append(m)
 f.append(a)
 f.append(b)
 f.append(c)
 f.append(d)
 f.append(e)
 f.append(p)
 f.append(g)
 f.append(h)
 f.append(i)
 f.append(j)
 f.append(k)

 return (f)
url= input("Enter URL:")
url1= predict_mal(url)
with open("model_pkl",'rb') as f_in:
 model=pickle.load(f_in)
 f_in.close()
url1= np.array(url1)
url1= url1.reshape(1, -1)
y_pred=model.predict(url1)
round(y_pred[0])
