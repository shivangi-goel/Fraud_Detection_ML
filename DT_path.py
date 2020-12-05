# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:20:26 2020

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
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 123)
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

from sklearn import tree
cfier=tree.DecisionTreeClassifier(max_leaf_nodes=12, max_depth=6)
cfier.fit(X_train, Y_train)

#getting the details of the fitted model, tree_ attribute gives the tree structure
n_nodes = cfier.tree_.node_count
children_left = cfier.tree_.children_left
children_right = cfier.tree_.children_right
feature = cfier.tree_.feature
threshold = cfier.tree_.threshold

node_indicator = cfier.decision_path(X_test)#retrieves the decision path of each datapoint

#getting id of the leaves
leaf_id = cfier.apply(X_test)

node_depth=np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves=np.zeros(shape=n_nodes, dtype=bool)
stack=[(0,0)]
#startingg with the root node i.e 0 and depth of root node is also 0
while(len(stack)>0):
    #using pop we traverse the nodes and make sure each node is traversed only once
    node_id,depth=stack.pop()
    node_depth[node_id]=depth
     
    #now checking the left and right child and adding their depths
    splitnode=children_left[node_id]!=children_right[node_id]
    #if they aren't equal we add them to the stack and increment the depth
    if splitnode:
        stack.append((children_left[node_id],depth+1))
        stack.append((children_right[node_id],depth+1))
    else:
        is_leaves[node_id]=True
print("The decision tree has %s nodes and the tree structure is as follows:" %(n_nodes))
for i in range(n_nodes):
    if(is_leaves[i]):
        print(" Node number %s at depth %s is a leaf node " %(i,node_depth[i]))
    else:
        print(" Node number %s at depth %s is a split node and since it is a binary tree we go to left node numbered %s if %s <= %s else we go to right node numbered %s" %(i, node_depth[i], children_left[i], X_test.columns[feature[i]],threshold[i], children_right[i]))

#finding path for a particular test row
#let us take id=0
eg_id=40
node_index = node_indicator.indices[node_indicator.indptr[eg_id]:
                                    node_indicator.indptr[eg_id + 1]]
    #datapoint goes through the nodes mentioned in the array
list1=[]
print('Rules used to predict the order number %s at row number %s are : ' % (X_test['SRC_ORD_NBR'].iloc[eg_id],eg_id))
for node_id in node_index:
    if(leaf_id[eg_id]==node_id):
        continue
    if(X_test.iloc[eg_id, feature[node_id]]<=threshold[node_id]):
        threshold_sign="<="
        list1.append(feature[node_id])
    else:
        threshold_sign=">"
        list1.append(feature[node_id])
    #print("decision id node %s : (X_test[%s,%s] (= %s) %s %s)" %(node_id, eg_id, feature[node_id], X_test.iloc[eg_id, feature[node_id]], threshold_sign, threshold[node_id]))
    print(" %s %s %s "%( X_test.columns[feature[node_id]],threshold_sign, threshold[node_id]))
from matplotlib import pyplot as plt
tree.plot_tree(cfier)
fig = plt.gcf()
fig.set_size_inches(150, 150)
plt.show()


feat_list=[]
for i in feature:
    if(i!=-2):
        feat_list.append(i)

feat_list_name=[]
for i in feat_list:
    feat_list_name.append(X.columns[i])
feat_list_name = list(dict.fromkeys(feat_list_name))

d={'Feature':feat_list_name, 'Row_number':feat_list}
row_list=pd.DataFrame(d)

Tag=[]
for i in row_list.Feature:
    if(i=='EMAIL_AGE_SCORE_VAL'):
        Tag.append('Email Details')
    elif(i=='VNDR_COMB_SCORE_VAL'):
        Tag.append('Product Details')
    elif(i=='SWPINN'):
        Tag.append('System Details')
    elif(i=='COM30'):
        Tag.append('System Details')
    elif(i=='IOIPR'):
        Tag.append('System Details')
    elif(i=='SWPENX'):
        Tag.append('System Details')
    elif(i=='PROD_QTY'):
        Tag.append('Product Details')
    elif(i=='EM128'):
        Tag.append('System Details')
row_list['Category']=Tag

#row_list.loc[row_list.Row_number==7, 'Category'].iloc[0]
new_row={'Feature':'EmailFirstSeenDataDiff', 'Row_number':159,'Category':'Email Details'}
row_list = row_list.append(new_row, ignore_index=True)
eg_id=152

node_index = node_indicator.indices[node_indicator.indptr[eg_id]:
                                    node_indicator.indptr[eg_id + 1]]
#datapoint goes through the nodes mentioned in the array
list1=[]
num=[]
i=0
print('Rules used to predict the order number %s at row number %s are : ' % (X_test['SRC_ORD_NBR'].iloc[eg_id],eg_id))
for node_id in node_index:
    if(leaf_id[eg_id]==node_id):
        continue
    if(X_test.iloc[eg_id, feature[node_id]]<=threshold[node_id]):
        threshold_sign="<="
        list1.append(feature[node_id])
        val=row_list.loc[row_list.Feature==X_test.columns[feature[node_id]], 'Category'].iloc[0]
        num.append(val)
    else:
        threshold_sign=">"
        list1.append(feature[node_id])
        val=row_list.loc[row_list.Feature==X_test.columns[feature[node_id]], 'Category'].iloc[0]
        num.append(val)
    print(" Category: %s| %s %s %s "%(num[i],X_test.columns[feature[node_id]],threshold_sign, threshold[node_id]))
    i=i+1
