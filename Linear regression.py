#!/usr/bin/env python
# coding: utf-8

# Name: Annissa Ajit Pereira
# 
# Email ID: annissa.p01@gmail.com
# 
# Github Profile: https://www.linkedin.com/in/annissa-pereira-b71a47208/
# 
# Linkedin Profile: https://github.com/annissapereira
# 
# Task 1: Prediction using Supervised ML GRIP
# 
# Used Linear Regression Model 
# 
# There are two features given in the dataset. Using hours features we have to predict the scores of the student.
# 
# Linear Regression with Python
# 
# 

# Import Libraries

# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Data

# In[3]:


student_data = pd.read_csv("http://bit.ly/w-data")


# Check out the Data

# In[4]:


student_data.head()


# Describing the data

# In[5]:


student_data.describe()


# Dimension Of The DataFrame

# student_data.shape

# Exploratory Data Analysis

# In[10]:


student_data.plot(x='Hours',y='Scores',style='.')  
plt.title('Hrs Studied vs Percentage Score')  
plt.xlabel('Hours Studied')  
plt.ylabel('% of student\s score')  
plt.show()


# Spliting the data input and output

# In[11]:


x = student_data.iloc[:,:-1].values
y = student_data.iloc[:,1].values


# Training a Linear Regression Model

# Let's now begin to train out regression model. We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the scores column.
# 

# Train Test Split

# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# Linear Regression

# In[14]:


from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(x_train,y_train)


# Visualising the training data

# In[18]:


plt.scatter(x_train,y_train,color = 'green')
plt.plot(x_train,linear_regression.predict(x_train),color='blue')
plt.title('Hours Studied vs Percentage Scored')
plt.xlabel("Hours student studied")
plt.ylabel('% of student\'s score')
plt.show()


# Prediction from our Model 

# In[19]:


y_pred = linear_regression.predict(x_test)
y_pred


# Model Evaluation

# Comparing the Actual and Predicted Values (in tabular form).

# In[25]:


df = pd.DataFrame({'Actual Value': y_test, 'Predicted value': y_pred})
df


# In[26]:


df.plot(kind="bar",figsize=(6,6))
plt.title('Hours Studied vs Percentage Scored')
plt.xlabel('Hours student studied')
plt.ylabel('% of student\'s score')
plt.show()


# Estimating training and test score

# In[28]:


print("Training Score:", linear_regression.score(x_train,y_train))
print("Test Score:", linear_regression.score(x_test,y_test))
      


# Predicted score if a student studies for 9.5 hours per day

# In[29]:


hrs = [9.25]
ans = linear_regression.predict([hrs])
print("Hours student study = {}".format(hrs))
print("Predicted Score of student= {}".format(ans[0]))


# Regression Evaluation Metrics

# Finding the residuals : It is very important to calculate the performance of the model.

# In[31]:


from sklearn import metrics
print('Mean Absolute Error :',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Square Error :', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error :',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# Conclusion
# 
# An approxiamte 93% is achieved by Student if he studies 9.25 hrs/day.
