#!/usr/bin/env python
# coding: utf-8

# # Movie Genre Classification using TF-IDF and Naive Bayes

# In[1]:


import opendatasets as od
od.download('https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[3]:


train_ds = open('genre-classification-dataset-imdb/train_data.txt', errors='ignore')


# In[4]:


test_ds = open('genre-classification-dataset-imdb/test_data.txt', errors='ignore')


# In[6]:


train_data = pd.read_csv(train_ds, sep=':::', names=['Title', 'Genre', 'Description'], engine='python')
print(train_data.describe())


# In[7]:


train_data.info()


# In[8]:


train_data.isnull().sum()


# In[9]:


test_data = pd.read_csv(test_ds, sep=':::', names=['Id', 'Title', 'Description'], engine='python')
test_data.head()


# # Data Visualization

# In[11]:


import seaborn as sns
# Plot the distribution of genres in the training data
plt.figure(figsize=(12,6))
sns.countplot(data=train_data, y='Genre', order=train_data['Genre'].value_counts().index, palette='icefire')
plt.xlabel('Count', fontsize=12, fontweight='bold')
plt.ylabel('Genre', fontsize=12, fontweight='bold')

# Plot the distribution of genres using a bar plot
plt.figure(figsize=(12,6))
counts = train_data['Genre'].value_counts()
sns.barplot(x=counts.index, y=counts, palette='icefire')
plt.xlabel('Genre', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Distribution of Genres', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, fontsize=12, fontweight='bold')
plt.show()


# # Data Preprocessing and Text Cleaning

# In[14]:


import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer


# In[15]:


# Initialize the stemmer and stop words
stemmer = LancasterStemmer()
stop_words = set(stopwords.words('english'))

# Define the clean_text function
def clean_text(text):
    text = text.lower()  # Lowercase all characters
    text = re.sub(r'@\S+', '', text)  # Remove Twitter handles
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z+']", ' ', text)  # Keep only characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')  # Keep words with length > 1 only
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')  # Remove stopwords
    text = " ".join([i for i in words if i not in stopwords and len(i) > 2])
    text = re.sub("\s[\s]+", " ", text).strip()  # Remove repeated/leading/trailing spaces
    return text

# Apply the clean_text function to the 'Description' column in the training and test data
train_data['Text_cleaning'] = train_data['Description'].apply(clean_text)
test_data['Text_cleaning'] = test_data['Description'].apply(clean_text)


# In[16]:


# Calculate the length of cleaned text
train_data['length_Text_cleaning'] = train_data['Text_cleaning'].apply(len)
# Visualize the distribution of text lengths
plt.figure(figsize=(8, 7))
sns.histplot(data=train_data, x='length_Text_cleaning', bins=20, kde=True, color='blue')
plt.xlabel('Length', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Distribution of Lengths', fontsize=16, fontweight='bold')
plt.show()


# # Text Vectorization Using TF-IDF

# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# In[18]:


tfidf_vectorizer = TfidfVectorizer()
X_train = tfidf_vectorizer.fit_transform(train_data['Text_cleaning'])
X_test = tfidf_vectorizer.transform(test_data['Text_cleaning'])


# # Training the model

# In[20]:


# Split the data into training and validation sets
X = X_train
y = train_data['Genre']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# # Accuracy score

# In[21]:


# Make predictions on the validation set
y_pred = classifier.predict(X_val)

# Evaluate the performance of the model
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_val, y_pred))


# # Make Predictions on the Test Data

# In[22]:


# Use the trained model to make predictions on the test data
X_test_predictions = classifier.predict(X_test)
test_data['Predicted_Genre'] = X_test_predictions


# In[23]:


# Save the test_data DataFrame with predicted genres to a CSV file
test_data.to_csv('predicted_genres.csv', index=False)

# Display the 'test_data' DataFrame with predicted genres
print(test_data)


# In[ ]:




