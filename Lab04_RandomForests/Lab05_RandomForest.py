#!/usr/bin/env python
# coding: utf-8

# # Random Forest
# 
# In this lab you will learn the most important aspects of the random forest learning method. 
# Completing this lab and analyzing the code will give you a deeper understanding of these type of models.
# In our experiments we will mostly use the package sklearn from which we import RandomForestClassifier.
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


from sklearn.datasets import make_classification, make_regression


# ## Data Creation
# 
# First of all, we create a data set containing 1000 samples with 2 features and two classes:

# In[ ]:


X, y = make_classification(n_samples = 1000,n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)


# <b>Exercise 1:</b>
# 
# Visualize the data set. It should look like this:
# <img src="figures/dataset.png" width="600"/>

# In[ ]:


### WRITE YOUR CODE HERE ###


# <b>Exercise 2:</b>
# 
# We split our data into train and test data. Then we can train our model (a random forest) on the train data and evaluate the model on the hold out test data. We split the data in a way that we train our model on 67% of the data and test our model on 33% of the data.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)


# Train a random forest on the training data and report the accuracy for this model on the train and test data using the default parameters of a random forest (from sklearn). What can you conclude from this? .

# In[ ]:


clf = RandomForestClassifier()
### WRITE YOUR CODE HERE ###


# ## Decision Boundary
# 
# Sometimes it is helpful to plot the decision boundary for a learned model. To do so, we create a grid of data points and calculate the probability of belonging to class 1. 
# (This can only be done once we have trained the random forest classifier)

# In[ ]:


x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
h = .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)


# Then we can plot the boundary using the 'contourf' function of matplotlib.

# In[ ]:


cm = plt.cm.RdBu  # color map
plt.contourf(xx, yy, Z, alpha=.8, cmap=cm)
colors = ['red','blue']
for cur_class in [0,1]:
    plt.scatter(X[y==cur_class, 0], X[y == cur_class, 1], c=colors[cur_class],
                       edgecolors='k', alpha=0.6, label=cur_class)
plt.legend()
plt.show()


# What can you conclude from the figure above?

# ## Parameter Selection
# 
# The implementation of the random forest algorithm in sklearn has many parameter. The most important ones are the number of trees used (n_estimators) and the maximal depth of a single tree (max_depth). Investigate how the number of used trees effects the training and testing accuracy.
# 
# <b>Exercise 3:</b>
# 
# Plot a diagram that shows the training and testing accuracy depending on the number of trees (from 1 to 20) used. This plot should look like this:
# <img src="figures/num_trees.png" width="600"/>

# In[ ]:


### WRITE YOUR CODE HERE ###


# <b>Optional:</b>
# 
# We want to further investigate how the number of used trees effects the training and testing accuracy. 
# We want to increase the number of trees to a maximum of 50.
# 
# Plot a diagram that shows the training and testing accuracy depending on the number of trees (from 1 to 50) used.

# In[ ]:


### WRITE YOUR CODE HERE ###


# <b>Exercise 4:</b>
# 
# We saw how the number of trees influences training and testing accuracy. 
# Now we do same thing for the other important parameter max_depth.
# Train multiple models with different depths. 
# The models should use 30 trees as estimators.
# 
# Plot a diagram that shows the training and testing accuracy depending on the maximal depth of a single tree (from 1 to 50). 
# This plot should look like this:
# <img src="figures/max_depth.png" width="600"/>

# In[ ]:


### WRITE YOUR CODE HERE ###


# ## Churn Data Set
# Lets revisit the churn data set from the first tutorial.

# In[ ]:


churn_df = pd.read_csv('telecom_churn.csv')
label = churn_df['Churn']
churn_df = churn_df.drop(columns=['Churn'])


# <b>Exercise 5:</b>
# 
# Create a data set containing only the numeric values. <b>Optional:</b> Try to convert all non numeric values to numeric values using a one hot encoding or by binning them. 

# In[ ]:


### WRITE YOUR CODE HERE ###


# <b>Exercise 6:</b>
# 
# Train a model on this data set and visualize the most important features in a figure. This should look like this (The scaling and order of features can be different):
# <img src="figures/importance.png" width="600"/>
# 
# <b>Hint</b>: The method feature_importance_ should be used.
# What can you conclude?

# In[ ]:


### WRITE YOUR CODE HERE ###


# <b>Exercise 7:</b>
# 
# If we want to use a random forest to solve regression problems we can use the RandomForestRegressor from sklearn.
# * Generate an easy regression data set using make_regression with 10 features. (use function make_regression)
# * Split the data set into a train and test set.
# * Train a model and report the training and testing mean square error (can be calculated using sklearn.metrics.mean_squared_error)

# In[ ]:


### WRITE YOUR CODE HERE ###

