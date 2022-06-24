#!/usr/bin/env python
# coding: utf-8

# # ANN on Loan Risk Predict
# 
# ***Before applying***
# 
# Please notice that, this is just a script for homework of *Commercial Bank*. All rights of codes reserved by Jason Wang from UESTC. 
# 
# Statistics are provided by teacher from course. What is used in this script is already normalized.
# 
# ***Before running***
# 
# Make sure the packages below is properly installed.
# 
# - numpy
# - pandas
# - scikit-learn

# ## Load and Split Raw Data

# In[1]:


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

train_test = pd.read_csv('./normalized_train_test.csv')
print('train_test shape: ' + str(train_test.shape))

# In[2]:


X = train_test.drop(['Unnamed: 0', 'ID', 'Default_Status'], axis=1)
Y = train_test['Default_Status']
print('X shape: ' + str(X.shape))
print('Y shape: ' + str(Y.shape))

# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

print('X_train shape: ' + str(X_train.shape))
print('X_test  shape: ' + str(X_test.shape))
print('Y_train shape: ' + str(Y_train.shape))
print('Y_test  shape: ' + str(Y_test.shape))

# ## Train ANN Model
# 
# The model here consists of two layers. One has 50 neurons while another has 20.

# In[4]:


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 20), random_state=1)
clf.fit(X_train, Y_train)

# ## Test the Model

# In[5]:


Y_test_gen = clf.predict(X_test)
Y_test = np.array(Y_test)
match = 0

for index in range(len(Y_test)):
    if Y_test[index] == Y_test_gen[index]:
        match += 1

matchRate = match / len(Y_test)

# In[6]:


print('The accuracy of this ANN method is ' + str(matchRate))

# ## Predict
# 
# The statistics used as basis to predict are also normalized.

# In[7]:


predict = pd.read_csv('./normalized_predict.csv')
print('predict shape: ' + str(predict.shape))

# In[8]:


X_predict = predict.drop(['Unnamed: 0', 'ID', 'Default_Status'], axis=1)
Y_predict = predict['Default_Status']
print('X_predict shape: ' + str(X_predict.shape))
print('Y_predict shape: ' + str(Y_predict.shape))

# In[9]:


Y_predict_gen = clf.predict(X_predict)
Y_predict_gen

# In[10]:


predict_result = predict.drop(['Unnamed: 0', 'Default_Status'], axis=1)
predict_result.insert(1, 'Default_Status', Y_predict_gen)
predict_result.to_csv('./predict_result.csv')
predict_result.head()

# In[ ]:
