#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import pickle
import streamlit as st
from PIL import Image


# In[2]:


url = 'https://github.com/stoozman/heart_diseases/blob/main/test.csv?raw=true'
data_test = pd.read_csv(url,index_col=0)
url2 = 'https://github.com/stoozman/heart_diseases/blob/main/train.csv?raw=true'
data_train = pd.read_csv(url2,index_col=0)


# In[3]:


data_train.info()


# In[4]:


data_train['gender'].unique()


# In[5]:


data_train['cholesterol'].unique()


# In[6]:


data_train['gluc'].unique()


# In[7]:


data_train['smoke'].unique()


# In[8]:


data_train['alco'].unique()


# In[9]:


data_train['active'].unique()


# In[10]:


data_train['age'] = data_train['age'].astype(float)
data_train['age'] = round(data_train['age']/365, 2)
data_test['age'] = data_test['age'].astype(float)
data_test['age'] = round(data_test['age']/365, 2)


# In[11]:


test_df = data_test


# In[12]:


data_train = data_train.drop('id', axis=1)
data_test = data_test.drop('id', axis=1)


# In[13]:


train = pd.get_dummies(data_train)
test = pd.get_dummies(data_test)
display(train.shape)
display(test.shape)


# In[14]:


data_train, data_valid= np.split(data_train.sample(frac=1, random_state=12345), [int(.621552287*len(data_train))])
[len(data_train), len(data_valid)]


# In[15]:



numeric = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo','cholesterol', 'gluc', 'smoke', 'alco', 'active']
scaler = StandardScaler()
features_train = data_train.drop(['cardio'], axis=1)
features_valid = data_valid.drop(['cardio'], axis=1)
features_test = data_test
target_train = data_train['cardio']
target_valid = data_valid['cardio']

scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])


# In[16]:


model = RandomForestClassifier(max_depth = 9, n_estimators = 24, random_state=12345)
model.fit(features_train, target_train)
predictions = model.predict(features_test)


# In[17]:


data_test['cardio'] = predictions


# In[18]:


data_test['cardio'].value_counts(normalize=True)


# In[19]:



pickle_out = open("classifier.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# In[ ]:





# In[ ]:




