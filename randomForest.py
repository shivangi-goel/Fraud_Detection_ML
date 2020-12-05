# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:36:58 2020

@author: DELL
"""

import numpy as np
import pandas as pd
dataset=pd.read_csv("C:\\Users\\DELL\\Desktop\\dell_work\\encoded_file.csv")
X=dataset.drop(['Y'],axis=1)
Y=dataset['Y']
#A=[]

# Splitting the dataset into the Training set and Test set
X=X.drop(['Unnamed: 0'],axis=1)
X=X.drop(['Unnamed: 0.1'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
model=classifier.fit(X_train, Y_train)
#model.summary()
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred)

feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
"""
print(feature_importances)
for i in range(0, len(feature_importances)):
    if(feature_importances['importance'][i]<0.001):
        A.append(feature_importances.index[i])
for i in range(0,len(A)):
    X=X.drop([A[i]],axis=1)
"""
    
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor

m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, Y_train)
#tree.plot_tree(m)
from IPython import display
str_tree = export_graphviz(m.estimators_[0], 
   out_file=None, 
   feature_names=X_test.columns, # column names
   filled=True,        
   special_characters=True, 
   rotate=True, 
   precision=1)
display.display(str_tree)

import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

#visualization for test data
from dtreeviz.trees import dtreeviz
from sklearn import tree
cfier=tree.DecisionTreeClassifier(max_depth=4)
cfier.fit(X_train, Y_train)
viz=dtreeviz(cfier,X_test.iloc[[1]], Y_train,feature_names=X_test.columns, target_name="Fraud", class_names=[0,1])

display.display(viz)
viz.view()
"""
X['Y']=Y
import plotly.express as px
fig = px.treemap(X, path=X.columns, values='Y')
fig.show()
"""
X2=X_test.iloc[[1]]

#getting the details of the fitted model, tree_ attribute gives the tree structure
n_nodes = cfier.tree_.node_count
children_left = cfier.tree_.children_left
children_right = cfier.tree_.children_right
feature = cfier.tree_.feature
threshold = cfier.tree_.threshold

node_indicator = cfier.decision_path(X_test)#retrieves the decision path of each datapoint

#getting id of the leaves
leaf_id = cfier.apply(X_test)

#finding path for a particular test row
#let us take id=0
eg_id=40
node_index = node_indicator.indices[node_indicator.indptr[eg_id]:
                                    node_indicator.indptr[eg_id + 1]]
    #datapoint goes through the nodes mentioned in the array
    
print('Rules used to predict sample %s: ' % eg_id)
for node_id in node_index:
    if(leaf_id[eg_id]==node_id):
        continue
    if(X_test.iloc[eg_id, feature[node_id]]<=threshold[node_id]):
        threshold_sign="<="
    else:
        threshold_sign=">"
    print("decision id node %s : (X_test[%s,%s] (= %s) %s %s)" %(node_id, eg_id, feature[node_id], X_test.iloc[eg_id, feature[node_id]], threshold_sign, threshold[node_id]))
node_id=0
if(leaf_id[eg_id]==node_id):
    print("not in tree")
if(X_test.iloc[eg_id, feature[node_id]]<=threshold[node_id]):
    threshold_sign="<="
else:
    threshold_sign=">"
print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)" %(node_id, eg_id, feature[node_id], X_test.iloc[eg_id, feature[node_id]], threshold_sign, threshold[node_id]))   