#!/usr/bin/env python
# coding: utf-8

# #Importing Liberaries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# #Loading Bankruptcy Dataset

# In[2]:


df=pd.read_excel('Bankruptcy (2).xlsx')
df


# In[ ]:





# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df['class'].value_counts()


# In[7]:


df.isnull().sum()


# #Exploratory Data Analysis

# In[8]:


df.plot(kind='box',subplots=True,layout=(3,2))


# In[9]:


df.plot(kind='hist',subplots=True,layout=(3,2))


# In[10]:


df.plot(kind='kde',subplots=True,layout=(3,2))


# In[11]:


sns.countplot(x='industrial_risk',data=df,palette='rainbow')


# In[12]:


pd.crosstab(df['industrial_risk'],df['class']).plot(kind='bar')


# In[13]:


sns.countplot(x='management_risk',data=df,hue='class')


# In[14]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['class']=le.fit_transform(df['class'])
df.head()


# In[15]:


corr=df.corr()
corr


# In[16]:


plt.figure(figsize=(8,8))
sns.heatmap(corr,annot=True)


# In[17]:


#import libraries for model building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[18]:


# Split into train-test sets
x=df.drop('class',axis=1)
y=df['class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[19]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# #Model Building

# In[20]:


model_lr=LogisticRegression()
model_lr.fit(x_train,y_train)
y_pred_lr=model_lr.predict(x_test)
print(accuracy_score(y_test,y_pred_lr))
acc_lr=accuracy_score(y_test,y_pred_lr)
model_lr.predict_proba(x_test)


# In[21]:


model_gb=GaussianNB()
model_gb.fit(x_train,y_train)
y_pred_gb=model_gb.predict(x_test)
print(accuracy_score(y_test,y_pred_gb))
acc_gb=accuracy_score(y_test,y_pred_gb)


# In[22]:


model_mn=MultinomialNB()
model_mn.fit(x_train,y_train)
y_pred_mn=model_mn.predict(x_test)
print(accuracy_score(y_test,y_pred_mn))
acc_mn=accuracy_score(y_test,y_pred_mn)


# In[23]:


model_svm=SVC()
model_svm.fit(x_train,y_train)
y_pred_svm=model_svm.predict(x_test)
print(accuracy_score(y_test,y_pred_svm))


# In[24]:


kernel=['linear','rbf','polynomial']
C=[5,10,12,15,16]
gamma=[50,5,10,6]
param_grid=dict(kernel=kernel,C=C,gamma=gamma)
grid=GridSearchCV(estimator=SVC(),param_grid=param_grid)
grid.fit(x_train,y_train)

print(grid.best_score_)
print(grid.best_params_)
acc_svm=grid.best_score_


# In[25]:


model_knn=KNeighborsClassifier(n_neighbors=5)
model_knn.fit(x_train,y_train)
y_pred_knn=model_knn.predict(x_test)
print(accuracy_score(y_test,y_pred_knn))


# In[26]:


n_neigh=np.arange(1,30)
param_grid=dict(n_neighbors=n_neigh)
grid=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=param_grid)
grid.fit(x_train,y_train)

print(grid.best_score_)
print(grid.best_params_)
acc_knn=grid.best_score_


# #Model Evaluation

# In[27]:


w=['LOGISTIC_REGRESSION','SVM','GAUSSIAN_NB','MULTINOMIAL_NB','KNN']
a=[acc_lr,acc_svm,acc_gb,acc_mn,acc_knn]
acc_data=pd.DataFrame({'Algorithms':w,'Accuracy':a})
acc_data.sort_values(by='Accuracy',ascending=False,inplace=True)
acc_data.set_index('Algorithms')


# In[28]:


#Choosing Best predecting model
accuracies = {'LOGISTIC_REGRESSION': acc_lr, 'SVM': acc_svm, 'GAUSSIAN_NB': acc_gb, 'MULTINOMIAL_NB': acc_mn, 'KNN': acc_knn} # Define the accuracies dictionary
best_model = max(accuracies, key=accuracies.get) # Changed min to max to find the best model (highest accuracy)
print(f'The best model is: {best_model} with accuracy = {accuracies[best_model]:.2f}')


# In[29]:


import pickle

# Saving the trained Logistic Regression classifier using pickle
with open('LOGISTIC_REGRESSION.pkl', 'wb') as file:
    pickle.dump(model_lr, file)

