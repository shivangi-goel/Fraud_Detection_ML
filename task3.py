# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 01:47:21 2020

@author: Shivangi_Goel
"""


import pandas as pd
import numpy as np
import csv
dataset=pd.read_csv("C:\\Users\\Shivangi_Goel\\Desktop\\python code\\file1.csv")
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
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

"""
Another way to encode the dataset when we have larger number of columns
mylist = list(dataset.select_dtypes(include=['object']).columns)
X_new=dataset
#X_new[mylist[0]]=lab_enc.fit_transform(X_new[mylist[0]])
for i in range(0,len(mylist)):
    X_new[mylist[i]]=lab_enc.fit_transform(X_new[mylist[i]])
X_new=dataset.drop(['FRAUD_FLAG'],axis=1)
"""

"""
#oversampling of data
from imblearn.over_sampling import SMOTE #Over sampling
sm = SMOTE(ratio='auto',kind='regular')
X_sampled,y_sampled = sm.fit_sample(X,Y.values.ravel())
#Percentage of fraudlent records in original data
Source_data_no_fraud_count = len(dataset[dataset.Class==0])
Source_data_fraud_count = len(dataset[dataset.Class==1])
print('Percentage of fraud counts in original dataset:{}%'.format((Source_data_fraud_count*100)/(Source_data_no_fraud_count+Source_data_fraud_count)))

#Percentage of fraudlent records in sampled data
Sampled_data_no_fraud_count = len(y_sampled[y_sampled==0])
Sampled_data_fraud_count = len(y_sampled[y_sampled==1])
print('Percentage of fraud counts in the new data:{}%'.format((Sampled_data_fraud_count*100)/(Sampled_data_no_fraud_count+Sampled_data_fraud_count)))


from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from sklearn.svm import LinearSVC
X_resample, Y_resample=SMOTE(random_state=42).fit_resample(X,Y)
print(sorted(Counter(Y_resample).items()))
clf_smote=LinearSVC().fit(X_resample,Y_resample)
#upsampling
#X_train['Y_train']=Y_train
"""

#upsampling
X_train['Y']=Y_train
from sklearn.utils import resample
df_maj=X_train[X_train.Y==0]
df_min=X_train[X_train.Y==1]
df_min_upsample=resample(df_min,replace=True,n_samples=len(df_maj),random_state=123)
frames=[df_maj,df_min_upsample]
df_upsampled=pd.concat(frames)
df_upsampled.Y.value_counts()

#predicting model
Y_new=df_upsampled.Y
X_new=df_upsampled.drop('Y',axis=1)
#train test split
X_new_train, X_new_test, Y_new_train, Y_new_test = train_test_split(X_new, Y_new, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
new_model=LogisticRegression().fit(X_new_train,Y_new_train)
pred_Y_new=new_model.predict(X_new_test)
#np.unique(pred_Y_new)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_new_test, pred_Y_new)
from sklearn.metrics import accuracy_score
accuracy_score(Y_new_test,pred_Y_new)
#accuracy of upsampled model is around 50.08%

#For random forest
from sklearn.ensemble import RandomForestClassifier
new_model2=RandomForestClassifier().fit(X_new_train,Y_new_train)
pred_Y_new2=new_model2.predict(X_new_test)
#np.unique(pred_Y_new)
accuracy_score(Y_new_test,pred_Y_new2)
cm2 = confusion_matrix(Y_new_test, pred_Y_new2)
#randomforest gives an accuracy of 99.95%


#downsampling
#df_maj, df_min already defined in upsample part
df_maj_downsample=resample(df_maj, replace=False,n_samples=len(df_min),random_state=123)
frames2=[df_maj_downsample,df_min]
df_downsampled=pd.concat(frames2)
df_downsampled.Y.value_counts()
#predicting model
Y_new2=df_downsampled.Y
X_new2=df_downsampled.drop('Y',axis=1)

X_new_train2, X_new_test2, Y_new_train2, Y_new_test2 = train_test_split(X_new2, Y_new2, test_size = 0.25, random_state = 0)

new_model_d=LogisticRegression().fit(X_new_train2,Y_new_train2)
pred_Y_new_d=new_model_d.predict(X_new_test2)
accuracy_score(Y_new_test2,pred_Y_new_d)
#accuracy is around 52.20%

#random forest
new_model_rd=RandomForestClassifier().fit(X_new_train2,Y_new_train2)
pred_Y_new_rd=new_model_rd.predict(X_new_test2)
#np.unique(pred_Y_new)
accuracy_score(Y_new_test2,pred_Y_new_rd)
#accuracy of random forest is again 92.51%
X['Y']=Y
X.to_csv("encoded_file.csv")