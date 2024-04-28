#!/usr/bin/env python
# coding: utf-8

# In[13]:


# importing preliminary libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('used_device_data.csv')


# In[3]:


from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Profiling Report")
profile


# In[4]:


df.head()


# In[5]:


df.info()


# In[ ]:


df.describe()


# In[7]:


# checking for null values in the dataset. 
# If there are nulls this could effect the prediction so we are going to fill null values with mode.
# this dataset has a lot of categorical data and filling with mode is the best option for categorical data.
# We will later impute these missing values.
print(df.isnull().sum())


# In[8]:


# Finding duplicates in the dataset
duplicates=df[df.duplicated()]


# In[9]:


# normally if there are duplicates then we will delete duplicates as this will cause overfitting of data.
# There are no duplicates in our case.
len(duplicates)


# In[15]:


# checking for correlation between data in the correlation matrix 
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
num_df = df[numerical_cols]
correlation_matrix = num_df.corr()
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')


# In[16]:


# understanding the distribution of categorical and numerical columns 
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[17]:


categorical_cols = df.select_dtypes(include=['object']).columns
for i in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=i, data=df, palette='pastel')
    plt.title(f'Count of Categories in {i}')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


# In[18]:


# understanding distribution of categorical and numerical columns will tell us if the data is skewed towards some values.
# We can see in the discribution that data is skewed towards android devices.
# Data has less entries for 5g mobiles.


# In[19]:


# filling missing values with mode as most of the data is categorical and has only certain set of values
# Loop through each column and fill missing values with mode
for col in df.columns:
    mode_val = df[col].mode()[0]  # Find the mode for the column
    df[col].fillna(mode_val, inplace=True)  # Fill missing values with the mode value

# Check if any missing values remain
print(df.isnull().sum())


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[21]:


# One hot encoding categorical columns with prefix
for i in categorical_cols:
    df = pd.concat([df, pd.get_dummies(df[i], prefix=i)], axis=1)
    df = df.drop(i, axis=1)


# In[22]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Normalize numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# In[23]:


from xgboost import XGBRegressor
# Assuming 'X' contains your feature columns and 'y' contains your target variable
X = df.drop('normalized_used_price', axis=1)
y = df['normalized_used_price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)


# Predictions
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Evaluate models
print("Linear Regression:")
print("Mean Squared Error:", mean_squared_error(y_test, lr_pred))
print("R-squared:", r2_score(y_test, lr_pred))

print("\nRandom Forest:")
print("Mean Squared Error:", mean_squared_error(y_test, rf_pred))
print("R-squared:", r2_score(y_test, rf_pred))



# In[24]:


import xgboost  as xgb
xgb_params = {
    'max_depth': 3, 
    'eta': 0.05, 
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',  
    'eval_metric': 'rmse',           
    'seed': 0
}

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)
evals = [(dtrain,'train'),(dtest,'eval')]
xgb_model = xgb.train (params = xgb_params,
              dtrain = dtrain,
              num_boost_round = 2000,
              verbose_eval=50, 
              early_stopping_rounds = 500,
              evals=evals,
              #feval = f1_score_cust,
              maximize = True)
 
# plot the important features  
fig, ax = plt.subplots(figsize=(6,9))
xgb.plot_importance(xgb_model,  height=0.8, ax=ax)
plt.show()

y_pred = xgb_model.predict(dtest)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)


# In[25]:


import shap

# Create a TreeExplainer object
explainer = shap.TreeExplainer(xgb_model)

# Calculate Shapley values for the test data
shap_values = explainer.shap_values(X_test)

# Plot the SHAP summary plot
shap.summary_plot(shap_values, X_test)


# In[27]:


# Create a TreeExplainer object
explainer_rf = shap.TreeExplainer(rf_model)

# Calculate Shapley values for the test data
shap_values_rf = explainer_rf.shap_values(X_test)

# Plot the SHAP summary plot
shap.summary_plot(shap_values_rf, X_test)


# In[ ]:





# In[ ]:




