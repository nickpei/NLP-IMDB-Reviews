#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import string
import nltk
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# load training results
with open(r'data_train_wostopNpunc.txt', 'r', encoding='UTF-8') as f:
    data_train = json.load(f)
first_1000 = data_train[:1000]
last_1000 = data_train[-1000:]
df = pd.DataFrame(first_1000 + last_1000).fillna(0)

### load test results
##with open(r'data_test_wostopNpunc.txt', 'r', encoding='UTF-8') as f:
##    data_test = json.load(f)
##first_1000 = data_test[:1000]
##last_1000 = data_test[-1000:]
##df = pd.DataFrame(first_1000 + last_1000).fillna(0)
### split
##features = df.drop(['__FileID__', '__CLASS__'], axis=1)
##labels = df.__CLASS__
##X_test, X_val, Y_test, Y_val = sklearn.model_selection.train_test_split(features, labels, test_size=0.8, 
##                                                                          random_state=1)
##print("\nTesting acc:", 
##      DecisionTree2.score(X_test, Y_test))


# ## Training/Validation Split

# In[3]:

features = df.drop(['__FileID__', '__CLASS__'], axis=1)
labels = df.__CLASS__
X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(features, labels, test_size=0.2, 
                                                                          random_state=1)
print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

#(1600, 28389) (400, 28389) (1600,) (400,)




 # Single Decision Tree

 In[4]:


DecisionTree = tree.DecisionTreeClassifier(criterion='entropy')
DecisionTree.fit(X_train, Y_train)
print("Training acc:", DecisionTree.score(X_train, Y_train), "\nValidation acc:", 
      DecisionTree.score(X_val, Y_val))
# Results:
# Training acc: 1.0 
# Validation acc: 0.685

# In[ ]:


parameters = {"max_depth": [None, 10, 100, 1000],
              "min_samples_split": [5, 10, 50, 100, 500, 1000],
              "min_samples_leaf": [10, 100, 1000],
              "max_leaf_nodes": [None, 10, 100, 1000],
              }
dt_search = GridSearchCV(DecisionTree, parameters)
dt_search.fit(X_train, Y_train)
print("The best parameters: " + str(dt_search.best_params_))

# The best parameters: {'max_depth': None, 'max_leaf_nodes': 1000,
# 'min_samples_leaf': 10, 'min_samples_split': 50}


 In[17]:


DecisionTree2 = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = None, max_leaf_nodes = 1000, 
                                            min_samples_leaf = 10, min_samples_split = 50)
DecisionTree2.fit(X_train, Y_train)
print("Training acc:", DecisionTree2.score(X_train, Y_train), "\nValidation acc:", 
      DecisionTree2.score(X_val, Y_val))

# Results:
# Training acc: 0.82 
# Validation acc: 0.6625



# ## Adaboost

# In[20]:


Boost = AdaBoostClassifier(base_estimator=DecisionTree2, n_estimators=100)
Boost.fit(X_train, Y_train)
print("Training acc:", Boost.score(X_train, Y_train), "\nValidation acc:",
      Boost.score(X_val, Y_val))

# Results:
# Training acc: 1.0 
# Validation acc: 0.845



RandomForest = RandomForestClassifier(criterion = 'entropy', n_estimators=100)
RandomForest.fit(X_train, Y_train)
print("Training acc:", RandomForest.score(X_train, Y_train), "\nValidation acc:",
      RandomForest.score(X_val, Y_val))

# Training acc: 1.0 
# Validation acc: 0.855


parameters = {"min_samples_split": [2, 5, 10, 20],
              "max_depth": [None, 2, 5, 10, 20],
              "min_samples_leaf": [1, 5, 10, 20],
              "max_leaf_nodes": [None, 5, 10, 20, 50],
              }
rfc_search = GridSearchCV(RandomForest, parameters)
rfc_search.fit(X_train, Y_train)
print("The best parameters: " + str(rfc_search.best_params_))

# The best parameters: {'max_depth': None, 'max_leaf_nodes': None,
# 'min_samples_leaf': 1, 'min_samples_split': 5}

RandomForest2 = RandomForestClassifier(criterion = "entropy", max_depth = None, max_leaf_nodes = None, 
                                            min_samples_leaf = 1, min_samples_split = 5)
RandomForest2.fit(X_train, Y_train)
print("Training acc:", RandomForest2.score(X_train, Y_train), "\nValidation acc:", 
      RandomForest2.score(X_val, Y_val))

# Training acc: 1.0 
# Validation acc: 0.88

# AdaBoost
Boost = AdaBoostClassifier(base_estimator=RandomForest2, n_estimators=100)
Boost.fit(X_train, Y_train)
print("Training acc:", Boost.score(X_train, Y_train), "\nValidation acc:",
      Boost.score(X_val, Y_val))

# Training acc: 1.0 
# Validation acc: 0.8725


 # SVM

 In[23]:


SVM = SVC(probability=True)
SVM.fit(X_train, Y_train)
print("Training acc:", SVM.score(X_train, Y_train), "\nValidation acc:", 
      SVM.score(X_val, Y_val))
# Results:
# Training acc: 0.988125 
# Validation acc: 0.8775

# In[ ]:


parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 0.005, 0.001], 'C': [0.5, 1, 1.5, 2, 4]},
              {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1]}]
svm_search = GridSearchCV(SVM, parameters, cv=5, scoring="roc_auc", n_jobs=4)
svm_search.fit(X_train, Y_train)
print("The best parameters: " + str(svm_search.best_params_))

# The best parameters: {'C': 4, 'gamma': 0.001, 'kernel': 'rbf'}

 In[ ]:


SVM2 = SVC(probability=True, kernel='rbf', C=4 ,gamma=0.001)
SVM2.fit(X_train, Y_train)
print("Training acc:", SVM2.score(X_train, Y_train), "\nValidation acc:", 
      SVM2.score(X_val, Y_val))

# Results:
# Training acc: 0.985625 
# Validation acc: 0.8975


 # Multiple Naive Bayes

 In[26]:


NaiveBayes = MNB()
NaiveBayes.fit(X_train, Y_train)
print("Training acc:", NaiveBayes.score(X_train, Y_train), "\nValidation acc:",
      NaiveBayes.score(X_val, Y_val))
# Results:
# Training acc: 0.99 
# Validation acc: 0.9325


 # SGD

 In[ ]:


sgd = SGD(max_iter=5, random_state=0,loss='modified_huber',n_jobs=4)
sgd.fit(X_train, Y_train)
print("Training acc:", sgd.score(X_train, Y_train), "\nValidation acc:",
      sgd.score(X_val, Y_val))
# Results:
# Training acc: 0.995 
# Validation acc: 0.88

# In[ ]:


parameters = {'alpha': [0.1, 0.5, 1, 1.5]}
sgd_search = GridSearchCV(sgd,parameters , scoring='roc_auc', cv=20)  
sgd_search.fit(X_train, Y_train)
print("The best parameters: " + str(sgd_search.best_params_))
# The best parameters: {'alpha': 0.1}

 In[ ]:


sgd2 = SGD(max_iter=5, random_state=0,loss='modified_huber',n_jobs=4,alpha=0.1)
sgd2.fit(X_train, Y_train)
print("Training acc:", sgd2.score(X_train, Y_train), "\nValidation acc:", 
      sgd2.score(X_val, Y_val))
# Results:
# Training acc: 0.99875 
# Validation acc: 0.9025


