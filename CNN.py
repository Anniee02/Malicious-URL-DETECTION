import pandas as pd
import numpy as np
import cv2
import math
from sklearn import model_selection
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
df=pd.read_csv("malicious_url.csv")
df.tail()
lst=df["url"]
def resizeList(chrs):
size=len(chrs)
sqr_rt=int(math.sqrt(size))+1
new_size=sqr_rt**2;
chrs=chrs+[0]*(new_size-len(chrs))
chrs=np.reshape(chrs,(sqr_rt,sqr_rt)).astype('float32')
chrs=cv2.resize(chrs,(7,7),interpolation=cv2.INTER_AREA)
return(chrs)
x=0
chars=[]
res = []
for ele in lst:
chars=[]
for num in ele:
if(ord(num)<255):
chars.append(ord(num))
else:
chars.append(255)
if(len(chars)<=49):
chars=chars+[0]*(49-len(chars))
chars=np.reshape(chars,(7,7))
else:
chars=resizeList(chars)
res.append(chars)
chr=df["url"][4]
chrs=[]
for num in chr:
chrs.append(ord(num))
print(resizeList(chrs))
df["conv_img"]=res
df.head()
df1=df[df["url_length"]>49]
df1.head()
df2=df[df["url_length"]<=49]
len(df2)
len(df1)
df["type_enc"] = pd.Categorical(pd.factorize(df.type)[0])
df.loc[df["type_enc"]==1,"type_enc1"]=int(0)
df.loc[df["type_enc"]!=1,"type_enc1"]=int(1)
df=df.astype({"type_enc1":"int32"})
df.head()
df2=df[df["url_length"]<=49]
df1=df[df["url_length"]>49]
x1=df1["conv_img"]
y1=df1["type_enc1"]
x2=df2["conv_img"]
y2=df2["type_enc1"]
xflat1=[]
xflat2=[]
for x in x1:
xflat1.append(x.flatten())
for x in x2:
xflat2.append(x.flatten())
X1_train, X1_test, Y1_train, Y1_test =
model_selection.train_test_split(xflat1,y1,test_size=0.33,random_state=42)
X2_train, X2_test, Y2_train, Y2_test =
model_selection.train_test_split(xflat2,y2,test_size=0.33,random_state=42)
X1_train=[x/255 for x in X1_train]
X1_test=[x/255 for x in X1_test]
X2_train=[x/255 for x in X2_train]
X2_test=[x/255 for x in X2_test]
X1_train=np.array(X1_train)
X1_test=np.array(X1_test)
X2_train=np.array(X2_train)
X2_test=np.array(X2_test)
X1_train=X1_train.reshape(X1_train.shape[0],*(7,7,1))
X2_train=X2_train.reshape(X2_train.shape[0],*(7,7,1))
X1_test=X1_test.reshape(X1_test.shape[0],*(7,7,1))
X2_test=X2_test.reshape(X2_test.shape[0],*(7,7,1))
model1 = models.Sequential()
model1.add(layers.Conv2D(32, (1, 1), activation='relu', input_shape=(7,7,1)))
#model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (1, 1), activation='relu'))
#model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (1, 1), activation='relu'))
model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(2, activation='softmax'))
model1.summary()
model1.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
model2 = models.Sequential()
model2.add(layers.Conv2D(32, (1, 1), activation='relu', input_shape=(7,7,1)))
#model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (1, 1), activation='relu'))
#model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (1, 1), activation='relu'))
model2.add(layers.Flatten())
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(2, activation='softmax'))
model2.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
history1 = model1.fit(X1_train, Y1_train, epochs=10,
batch_size=300,validation_data=(X1_test, Y1_test))
history2 = model2.fit(X2_train, Y2_train, epochs=10,
batch_size=300,validation_data=(X2_test, Y2_test))
model1.save("greater_model")
model2.save("lesser_model")


import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import math
def resizeList(chrs):
size=len(chrs)
sqr_rt=int(math.sqrt(size))+1
new_size=sqr_rt**2;
chrs=chrs+[0]*(new_size-len(chrs))
chrs=np.reshape(chrs,(sqr_rt,sqr_rt)).astype('float32')
chrs=cv2.resize(chrs,(7,7),interpolation=cv2.INTER_AREA)
return(chrs)
def getImg(url):
chars=[]
for num in url:
if(ord(num)<255):
chars.append(ord(num))
else:
chars.append(255)
if(len(chars)<=49):
chars=chars+[0]*(49-len(chars))
chars=np.reshape(chars,(7,7))
else:
chars=resizeList(chars)
return chars;
def getPrediction(test_url,model1,model2):
length=len(test_url)
test_url=getImg(test_url)
test_url=test_url/255
test_url=test_url.reshape(1,*(7,7,1))
prediction=0
if(length<=49):
prediction=model2.predict(test_url)
else:
prediction=model1.predict(test_url)
classes = np.argmax(prediction, axis = 1)
return(classes)
model1=tf.keras.models.load_model("greater_model")
model2=tf.keras.models.load_model("lesser_model")
print(getPrediction("https://www.g0oadsfasdgfgasgasdle.com/",model1,model2)
)
