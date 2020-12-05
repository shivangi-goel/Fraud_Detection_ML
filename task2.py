# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:04:36 2020

@author: Shivangi_Goel
"""
import pandas as pd
import numpy as np
import csv
dataset=pd.read_csv("C:\\Users\\Shivangi_Goel\\Desktop\\python code\\file1.csv")
dataset2=pd.read_csv("C:\\Users\\DELL\\Desktop\\dell_work\\ML_Data.csv")
dataset.dtypes
#dataset.head()
"""
from sklearn .preprocessing import OneHotEncoder
OneHotEncoder(handle_unknown='ignore').fit_transform(dataset)
X=dataset.iloc[:,2]
"""
Y=dataset.iloc[:,157].values
X=dataset.drop(['FRAUD_FLAG'],axis=1)
from sklearn.preprocessing import LabelEncoder
lab_enc =LabelEncoder()
Y = lab_enc.fit_transform(Y)
#X.iloc[:,[2]]=lab_enc.fit_transform(X.iloc[:,[2]])  
list=[3,9,10,11,12,14,15,17,22,23,24,25,26,27,28,29,30,31,34,36,38,39,40,41,43,157,158,159,160]
for i in list:
    X.iloc[:,[i]]=lab_enc.fit_transform(X.iloc[:,[i]]) 

"""
Another way to encode the dataset when we have larger number of columns
mylist = list(dataset.select_dtypes(include=['object']).columns)
X_new=dataset
#X_new[mylist[0]]=lab_enc.fit_transform(X_new[mylist[0]])
for i in range(0,len(mylist)):
    X_new[mylist[i]]=lab_enc.fit_transform(X_new[mylist[i]])
X_new=dataset.drop(['FRAUD_FLAG'],axis=1)
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
#cm--> TP=65206, FN=468, FP=92, TP=369
accuracy=(65206+369)/(65206+369+92+468)
#accuracy for logistic model is around 99.15%

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred)
#accuracy for logistic model is around 99.15%

#Analysing the test results
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm,cmap = ListedColormap(('red', 'green')))
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()

# 1 is T i.e fraud true and 0 is N i.e . not fraud


"""
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred2 = classifier.predict(X_test)
accuracy_score(Y_test, Y_pred2)
"""
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred3 = classifier.predict(X_test)
cm3= confusion_matrix(Y_test, Y_pred3)
accuracy_score(Y_test, Y_pred3)
#accuracy is around 99.2%
#cm3--> TP=65254, FP=479, FN=44, TN=358
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm3,cmap = ListedColormap(('red', 'green')))
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm3[i, j], ha='center', va='center', color='white')
plt.show()



