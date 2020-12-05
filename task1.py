import pandas as pd
import numpy as np
import csv
dataset=pd.read_csv("C:\\Users\\Shivangi_Goel\\Desktop\\python code\\df_with_ohe.csv")
dataset.head()
#dataset.dropna()

dataset.dropna(subset=['VNDR_COMB_SCORE_VAL'], inplace=True)
dataset.dropna(subset=['EMAIL_AGE_SCORE_VAL'], inplace=True)
dataset.shape
dataset[dataset.BILT_ADDR_VRFCTN_RSLT_CD !='**']
index_Name=dataset[dataset.BILT_ADDR_VRFCTN_RSLT_CD =='**'].index
dataset.drop(index_Name,inplace=True)
index_Name2=dataset[dataset.ECOMM_CD=='**'].index
dataset.drop(index_Name2,inplace=True)
matr=dataset.corr()
dataset.to_csv('file1.csv')
dataset2=pd.read_csv("C:\\Users\\Shivangi_Goel\\Desktop\\python code\\file1.csv")
dataset2.dropna(inplace=True)
del dataset2['Unnamed: 0']
dataset2.to_csv('file1.csv')


#Applying classification models
Y=dataset2.iloc[:,156].values
X=dataset2.iloc[:,[3,4,5,6,7]]

#Label encoding
from sklearn.preprocessing import LabelEncoder
lab_enc =LabelEncoder()
Y = lab_enc.fit_transform(Y)

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
accuracy=(65209+135)/(65209+135+89+702)
#accuracy for logistic model is around 98.80%



# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred2 = classifier.predict(X_test)

# Making the Confusion Matrix
cm2 = confusion_matrix(Y_test, Y_pred2)
accuracy2=(65298)/(65298+837)
#accuracy for SVM model is around 98.73%



# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred3 = classifier.predict(X_test)

cm3 = confusion_matrix(Y_test, Y_pred3)
accuracy3=(65173+342)/(65173+342+125+495)
#accuracy of Random Forest is 99.06%
