#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[8]:


df = pd.read_csv('spam.csv', encoding='ISO-8859-1')


# In[9]:


df.sample(5)


# In[10]:


df.shape


# ## 1. Data Cleaning

# In[11]:


df.info()


# In[12]:


# drop last 3 cols
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[13]:


df.sample(5)


# In[14]:


# renaming the cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[15]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[16]:


df['target'] = encoder.fit_transform(df['target'])


# In[17]:


df.head()


# In[18]:


# missing values
df.isnull().sum()


# In[19]:


# check for duplicate values
df.duplicated().sum()


# In[20]:


# remove duplicates
df = df.drop_duplicates(keep='first')


# In[21]:


df.duplicated().sum()


# In[22]:


df.shape


# ## 2.EDA

# In[23]:


df.head()


# In[24]:


df['target'].value_counts()


# In[25]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# ### Data is imbalanced

# In[26]:


import nltk


# In[27]:


nltk.download('punkt')


# In[28]:


df['num_characters'] = df['text'].apply(len)


# In[29]:


df.head()


# In[30]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[31]:


df.head()


# In[32]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[33]:


df.head()


# In[34]:


df[['num_characters','num_words','num_sentences']].describe()


# In[35]:


# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[36]:


#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[37]:


import seaborn as sns


# In[38]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[39]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[40]:


sns.pairplot(df,hue='target')


# ## 3. Data Preprocessing
# - Lower case
# - Tokenization
# - Removing special characters
# - Removing stop words and punctuation
# - Stemming

# In[50]:


from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer as ps


# In[51]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[54]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[ ]:


df['text'][10]


# In[53]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')


# In[55]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[56]:


df.head()


# In[59]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[60]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[61]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[62]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[63]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[64]:


df.head()


# In[65]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
        


# In[66]:


len(spam_corpus)


# In[70]:


from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[71]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[72]:


len(ham_corpus)


# In[74]:


from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0],y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[75]:


# Text Vectorization
# using Bag of Words
df.head()


# ## 4. Model Building

# In[76]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[77]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[ ]:


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)


# In[ ]:


# appending the num_character col to X
#X = np.hstack((X,df['num_characters'].values.reshape(-1,1)))


# In[78]:


X.shape


# In[79]:


y = df['target'].values


# In[80]:


from sklearn.model_selection import train_test_split


# In[81]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[82]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[83]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[84]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[85]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[86]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[ ]:


# tfidf --> MNB


# In[89]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[90]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[91]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[92]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[93]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[96]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[97]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[98]:


performance_df


# In[99]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[100]:


performance_df1


# In[101]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[102]:


# model improve
# 1. Change the max_features parameter of TfIdf


# In[103]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[104]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[105]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[106]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[107]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[108]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[109]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[110]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[111]:


voting.fit(X_train,y_train)


# In[112]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[113]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[114]:


from sklearn.ensemble import StackingClassifier


# In[115]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[116]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# ### After all of the experiment , we can see that Naive Bayes classifier is providing the best accuracy of 0.97 and best precision call of 1.0 so we will choose Naive Bayes classifier to predict this model.
