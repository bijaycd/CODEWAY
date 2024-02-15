#!/usr/bin/env python
# coding: utf-8

# In[2]:


import opendatasets as od
od.download('https://www.kaggle.com/datasets/kartik2112/fraud-detection')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[4]:


train_df = pd.read_csv('fraud-detection/fraudTrain.csv')
train_df


# In[5]:


test_df = pd.read_csv('fraud-detection/fraudTest.csv')
test_df.head()


# In[6]:


print(train_df.shape)
print(test_df.shape)


# In[7]:


train_df.columns


# ## About Dataset
# 
# This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.
# 
# ## Source of simulation
# 
# This was generated using Sparkov Data Generation | Github tool created by Brandon Harris. This simulation was run for the duration - 1 Jan 2019 to 31 Dec 2020. The files were combined and converted into a standard format.

# ## Column description
# 
# - `index` - Unique Identifier for each row
# - `trans_date_trans_time` - Transaction DateTime
# - `cc_num` - Credit Card Number of Customer
# - `merchant` - Merchant Name
# - `category` - Category of Merchant
# - `amt` - Amount of Transaction
# - `first` - First Name of Credit Card Holder
# - `last` - Last Name of Credit Card Holder
# - `gender` - Gender of Credit Card Holder
# - `street` - Street Address of Credit Card Holder
# - `city` - City of Credit Card Holder
# - `state` - State of Credit Card Holder
# - `zip` - Zip of Credit Card Holder
# - `lat` - Latitude Location of Credit Card Holder
# - `long` - Longitude Location of Credit Card Holder
# - `city_pop` - Credit Card Holder's City Population
# - `job` - Job of Credit Card Holder
# - `dob` - Date of Birth of Credit Card Holder
# - `trans_num` - Transaction Number
# - `unix_time` - UNIX Time of transaction
# - `merch_lat` - Latitude Location of Merchant
# - `merch_long` - Longitude Location of Merchant
# - `is_fraud` - Fraud Flag <--- `Target Class`

# ## Data Preprocessing

# In[6]:


train_df.info()


# In[7]:


test_df.info()


# In[9]:


train_df.isnull().sum()


# In[28]:


train_df.duplicated(['trans_num']).sum()


# In[29]:


train_df.duplicated(['unix_time']).sum()


# In[30]:


train_df.duplicated(['trans_date_trans_time']).sum()


# In[8]:


np.round(train_df.describe(), 2)


# In[9]:


np.round(train_df['amt'].describe(),2)


# In[34]:


train_df['amt'].plot(kind='box')


# In[36]:


train_df[train_df['amt']>15000]


# In[37]:


train_df['amt'].plot(kind='kde')


# ## Issuses with the dataset
# 
# - Incorrect datatype of `trans_date_trans_time`, `dob`, `unix_time`

# ## EDA

# In[18]:


train_df.drop(columns=['Unnamed: 0'], inplace=True)


# In[19]:


train_df


# In[21]:


train_df.rename(columns={"trans_date_trans_time":"transaction_time",
                         "cc_num":"credit_card_number",
                         "amt":"amount(usd)",
                         "trans_num":"transaction_id"},
                inplace=True)


# In[22]:


train_df["transaction_time"] = pd.to_datetime(train_df["transaction_time"], infer_datetime_format=True)
train_df["dob"] = pd.to_datetime(train_df["dob"], infer_datetime_format=True)


# In[24]:


from datetime import datetime

# Apply function utcfromtimestamp and drop column unix_time
train_df['time'] = train_df['unix_time'].apply(datetime.utcfromtimestamp)
train_df.drop('unix_time', axis=1)

# Add cloumn hour of day
train_df['hour_of_day'] = train_df.time.dt.hour


# In[26]:


train_df[['time','hour_of_day']]


# In[27]:


train_df.credit_card_number = train_df.credit_card_number.astype('category')
train_df.is_fraud =train_df.is_fraud.astype('category')
train_df.hour_of_day = train_df.hour_of_day.astype('category')


# In[28]:


train_df


# In[29]:


train_df.info()


# In[31]:


train_df['is_fraud'].value_counts()


# In[36]:


train_df['is_fraud'].value_counts().plot(kind='pie',autopct='%0.1f%%')
plt.legend()


# In[44]:


x = train_df['job'].value_counts().nlargest(10)
x


# In[45]:


x.plot(kind='bar')


# In[52]:


y = train_df['category'].value_counts()
y


# In[53]:


y.plot(kind='bar')


# In[56]:


pd.crosstab(train_df['is_fraud'],train_df['category'],normalize='columns')*100


# In[59]:


pd.crosstab(train_df['job'],train_df['is_fraud'],normalize='columns')*100


# In[60]:


pd.crosstab(train_df['is_fraud'],train_df['hour_of_day'],normalize='columns')*100


# In[61]:


sns.heatmap(pd.crosstab(train_df['is_fraud'],train_df['hour_of_day'],normalize='columns')*100)


# In[57]:


sns.heatmap(pd.crosstab(train_df['is_fraud'],train_df['category'],normalize='columns')*100)


# ## Train the model

# In[66]:


features = ['transaction_id', 'hour_of_day', 'category', 'amount(usd)', 'merchant', 'job']

#
X = train_df[features].set_index("transaction_id")
y = train_df['is_fraud']

print('X shape:{}\ny shape:{}'.format(X.shape,y.shape))


# In[67]:


from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(dtype=np.int64)
enc.fit(X.loc[:,['category','merchant','job']])

X.loc[:, ['category','merchant','job']] = enc.transform(X[['category','merchant','job']])


# In[68]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# In[70]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print('X_train shape:{}\ny_train shape:{}'.format(X_train.shape,y_train.shape))
print('X_test shape:{}\ny_test shape:{}'.format(X_test.shape,y_test.shape))


# In[97]:


from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report)


# ## Prediction and accuracy

# In[99]:


print(classification_report(y_test,y_pred))


# In[101]:


from sklearn.metrics import confusion_matrix
plt.figure(figsize=(8, 6))
cfs_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cfs_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

