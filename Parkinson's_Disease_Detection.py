#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# In[33]:


dataset=pd.read_csv("Parkinson's_Disease_Detection.csv")


# In[34]:


dataset.info()


# In[35]:


dataset.head()


# In[36]:


# checking the null values
dataset.isnull().sum()


# In[37]:


dataset['status'].value_counts()


# In[38]:


import matplotlib.pyplot as plt
dataset.hist(figsize = (16,20))


# In[39]:


plt.boxplot(dataset["PPE"])
plt.show()


# In[40]:


print(dataset['PPE'].quantile(0.10))
print(dataset['PPE'].quantile(0.90))


# In[41]:


print(dataset['PPE'].skew())
dataset['PPE'] = np.where(dataset['PPE']< 0.1 ,0.1,dataset['PPE'])
dataset['PPE'] = np.where(dataset['PPE']> 0.3 ,0.3,dataset['PPE'])
print(dataset['PPE'].skew())


# In[42]:


plt.boxplot(dataset["PPE"])
plt.show()


# In[43]:


# dividing data into features and labels
features = dataset.drop(columns=['name','status'], axis=1)
labels= dataset['status']


# In[44]:


#Normalizing data using min max scaler
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels


# In[45]:


# spiliting data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[46]:


model1 =SVC()
model1.fit(x_train,y_train)


# In[47]:


print('Accuracy of train data :',model1.score(x_train,y_train)*100,'%')


# In[48]:


print('Accuracy of test data :',model1.score(x_test,y_test)*100,'%')


# In[49]:


model2=KNeighborsClassifier(n_neighbors=5)


# In[50]:


model2.fit(x_train,y_train)


# In[51]:


print('Accuracy of train data :',model2.score(x_train,y_train)*100,'%')


# In[52]:


print('Accuracy of test data :',model2.score(x_test,y_test)*100,'%')


# In[53]:


model3=LogisticRegression()


# In[54]:


model3.fit(x_train,y_train)


# In[55]:


print('Accuracy of train data :',model3.score(x_train,y_train)*100,'%')


# In[56]:


print('Accuracy of test data :',model3.score(x_test,y_test)*100,'%')


# In[ ]:





# In[ ]:




