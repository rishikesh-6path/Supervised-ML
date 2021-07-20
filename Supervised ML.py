#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("ML.Dataset.txt")                                  


# In[22]:


df


# In[4]:


df.columns


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


def null_detection(df):
    num_cols = []
    
    count = 0
    t = []
    for i in num_cols:
        z = np.abs(stats.zscore(df[i]))
        for j in range(len(z)):
            if z[j]>3 or z[j]<-3:
                t.append(j)
                count+=1
    df = df.drop(list(set(t)))
    df = df.reset_index()
    df = df.drop('index', axis=1)
    print(count)
    return df


# In[9]:


df = null_detection(df)


# In[10]:


plt.rcParams["figure.figsize"] = [10,5]
df.plot(kind='scatter', x='Hours', y='Scores',style='.',color='orange',)
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.grid()
plt.show()


# In[11]:


df.corr(method = 'pearson')


# In[12]:


df.corr(method = 'spearman')


# In[13]:


X = df.iloc[:, :1].values
Y = df.iloc[:, 1:].values


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 ,random_state=0)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train , Y_train)


# In[15]:


line = model.coef_*X + model.intercept_

plt.rcParams["figure.figsize"] = [10,5]
plt.scatter(X_train , Y_train,color= 'orange')
plt.plot(X, line , color= 'black');
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.grid()
plt.show()


# In[16]:


print(X_test)
y_pred = model.predict(X_test)


# In[17]:


Y_test


# In[18]:


y_pred


# In[19]:


comp= pd.DataFrame({'Actual':[Y_test],'Predictd':[y_pred]})
comp


# In[20]:


hours = 7.5
my_pred = model.predict([[hours]])
print("The predicted score if a person sudies for", hours, "hours is", my_pred[0])


# In[21]:


from sklearn import metrics

print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




