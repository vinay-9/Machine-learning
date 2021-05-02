#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


df= pd.read_csv("Car_Sales.csv")
df.head()
df.isnull()
#dividing in the formof dependent and independent variables
X= df.iloc[:,0:1]
y=df.iloc[:,2:3]
print("Pair plotting")
sns.pairplot(df)

df.corr() ## how features are related with each other


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.4, random_state=0)

regressor= LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_test, y_test) # checking the R square value

#cross validation
score= cross_val_score(regressor, X, y, cv=2)
score.mean()

regressor.coef_   ##(intercepts in all the cases)
coef_df= pd.DataFrame(regressor.coef_, X.columns, columns=["coefficient"])
coef_df
# shows the unit increase in each parameters, what will be the effect on the outp

prediction=regressor.predict(X_test)
prediction
sns.distplot(prediction)
plt.scatter(y_test, prediction)
Sample_test=pd.DataFrame([[2020]])

Prediction_sample= regressor.predict(Sample_test)
Prediction_sample


# In[30]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


df= pd.read_csv("Car_Sales.csv")
df.head()
df.isnull()
#dividing in the formof dependent and independent variables
X= df.iloc[:,0:1]
y=df.iloc[:,2:3]


# In[31]:


df.isnull().values.any()


# In[37]:


df.head()


# In[33]:


X


# In[34]:


y


# In[35]:


plt.plot(X,y)
plt.title("trend graph")
plt.xlabel("year")
plt.ylabel("sales revenue")


# In[36]:


print("Pair plotting")
sns.pairplot(df)

df.corr() ## how features are related with each other


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.4, random_state=0)

regressor= LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_test, y_test) # checking the R square value

#cross validation
score= cross_val_score(regressor, X, y, cv=2)
score.mean()

regressor.coef_   ##(intercepts in all the cases)
coef_df= pd.DataFrame(regressor.coef_, X.columns, columns=["coefficient"])
coef_df
# shows the unit increase in each parameters, what will be the effect on the outp

prediction=regressor.predict(X_test)
prediction
sns.distplot(prediction)
plt.scatter(y_test, prediction)
Sample_test=pd.DataFrame([[2020]])

Prediction_sample= regressor.predict(Sample_test)
Prediction_sample


# In[ ]:




