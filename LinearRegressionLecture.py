#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[5]:


x = np.random.normal(1000,500,100)


# In[8]:


maindf = pd.read_csv('sampledatagen.csv')


# In[21]:


z = [3,5,3,5,7]
zmul = 12 * z
type(zmul)


# In[17]:


x = np.random.normal(1000,500,100)
type(x)


# In[22]:


y = 12*x+ 19


# In[37]:


maindf2 = pd.DataFrame([x, y]).T
maindf2.columns = ['X', 'Y']
maindf2.head()


# In[41]:


maindf3 = pd.DataFrame()
maindf3['X'] = x
maindf3['Y'] = y
maindf3.head()


# In[42]:


import matplotlib as mplt
import seaborn as sns


# In[45]:


mplt.pyplot.hist(maindf3['X'], bins = 10)


# In[48]:


linspacex = np.linspace(500, 1500, 100)
mplt.pyplot.hist(linspacex, bins = 10)


# In[47]:


sns.pairplot(maindf3)


# In[ ]:


pip install sklearn


# In[55]:


from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import 


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(maindf3['X']), maindf3['Y'], test_size=0.25)
print(type(X_train))
print(type(y_train))


# In[88]:


lmodelfirst = linear_model.LinearRegression()
lmodelfirst


# In[90]:


lmodelfirst.fit(X_train, y_train)


# In[93]:


ypredicted = lmodelfirst.predict(X_test)


# In[98]:


compareoutput = pd.DataFrame([ypredicted, y_test]).T
compareoutput.columns = ['Predicted', 'TestValue']

