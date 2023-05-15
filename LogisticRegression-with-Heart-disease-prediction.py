#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


#loading the csv data to pandas data frame
heart_data = pd.read_csv('Heart_Disease_Prediction.csv')


# In[3]:


#print first four rows
heart_data.head(4)


# In[4]:


#print last six rows
heart_data.tail(6)


# In[5]:


#number of rows
len(heart_data)


# In[6]:


#getting some info about the data
heart_data.info(verbose=False)


# In[7]:


#checking for missing values
heart_data.isnull().sum()


# In[8]:


#checking the distribution of target variable
heart_data['Heart Disease'].value_counts()


# Presence represents patients with heart defects Absence represents patients without heart defects

# In[13]:


heart_data.agg(
    {
        "Age": ["min", "max", "skew", "mean"],
        "Cholesterol": ["min", "max", "median", "mean"],
        "Chest pain type": ["min", "max", "median", "mean"],
     "BP"	: ["min", "max", "median", "skew", "mean"],	
     "FBS over 120"	: ["min", "max", "median", "skew", "mean"],
     "EKG results" : ["min", "max", "median", "skew", "mean"],
     	"Max HR" : ["min", "max", "median", "skew","mean"],
       	"Exercise angina"	: ["min", "max", "median", "skew","mean"],
     "ST depression" : ["min", "max", "median", "skew","mean"],	
     "Slope of ST": ["min", "max", "median", "skew","mean"],
     "Number of vessels fluro" : ["min", "max", "median", "skew","mean"],
     "Thallium"	: ["min", "max", "median", "skew","mean"],
     "Heart Disease": ["min", "max", "median", "skew","mean"]
    }
)


# Spliting the features and targets
# 

# In[9]:


X = heart_data.drop(columns='Heart Disease', axis=1)
Y = heart_data['Heart Disease']
print(X)


# In[10]:


print(Y)


# In[11]:


# Splitting the data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape) 


# In[12]:


model = LogisticRegression()


# Model Training

# In[13]:


model.fit(X_train, Y_train) 


# Accuracy score

# In[14]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[15]:


print(' training data accuracy :', training_data_accuracy)


# In[16]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[17]:


print('test data accuracy :', test_data_accuracy)


# Building a predictive system

# In[21]:


input_data= (48,1,4,122,222,0,2,186,0,0,1,0,3)

#change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

#reshape the numpy array to only predict for one instance,so our machine won't predict for all instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)


# In[ ]:




