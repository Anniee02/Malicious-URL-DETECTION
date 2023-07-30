import pandas as pd
import sklearn
from sklearn import svm, preprocessing
db = pd.read_csv("url_features.csv", index_col=0)
db.head()
db=sklearn.utils.shuffle(db)
db.head()
print('The shape of our dataset is:', db.shape)
header=["Number","url","type","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10"
,"f11","f12","f13"]
db.columns=header
db.drop(["Number","url"],axis=1,inplace=True)
db.head()
enc=preprocessing.LabelEncoder()
db["f1"]=enc.fit_transform(db["f1"])
db["f2"]=enc.fit_transform(db["f2"])
db["f3"]=enc.fit_transform(db["f3"])
db["f4"]=enc.fit_transform(db["f4"])
db["f5"]=enc.fit_transform(db["f5"])
db["f6"]=enc.fit_transform(db["f6"])
db["f7"]=enc.fit_transform(db["f7"])
db["f8"]=enc.fit_transform(db["f8"])
db["f9"]=enc.fit_transform(db["f9"])
db["f10"]=enc.fit_transform(db["f10"])
db["f11"]=enc.fit_transform(db["f11"])
db["f12"]=enc.fit_transform(db["f12"])
db["f13"]=enc.fit_transform(db["f13"])
db["type"]=enc.fit_transform(db["type"])
db.head()
X=db[["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12"]]
Y=db["type"]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
random_state=0)
import numpy as np
from sklearn.model_selection import cross_validate
X
Y
X_train
Y_train
import matplotlib.pyplot as plt
plt.style.use('classic')
from sklearn import neighbors
clf= neighbors.KNeighborsClassifier()
clf.fit(X_train,Y_train)
accuracy = clf.score(X_test, Y_test)
print('Model accuracy =', accuracy*100)
