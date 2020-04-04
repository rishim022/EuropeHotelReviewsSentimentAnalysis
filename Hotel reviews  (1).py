#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()


# In[3]:


data_hotel_reviews = pd.read_csv("Hotel_Reviews.csv")


# In[172]:


data_hotel_reviews.Hotel.head()


# In[5]:


for i in data_hotel_reviews.columns:
    print(i)


# In[6]:


len(data_hotel_reviews.Hotel_Name.unique())


# In[7]:


data_hotel_reviews.Reviewer_Nationality.describe()


# In[8]:


Reviewer_Nat_Count = data_hotel_reviews.Reviewer_Nationality.value_counts()
print(Reviewer_Nat_Count[:10])


# In[9]:


data_hotel_reviews_plot = data_hotel_reviews[["Hotel_Name","Average_Score"]].drop_duplicates()

sns.set(font_scale = 2.2)
a4_dims = (22,13)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.countplot(ax = ax,x = "Average_Score",data=data_hotel_reviews)


# In[11]:


text = " "
for i in range(data_hotel_reviews.shape[0]):
    text = " ".join([text,data_hotel_reviews["Reviewer_Nationality"].values[i]])


# In[12]:


from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white', width = 500,                      height=200, max_font_size=50, max_words=50).generate(text)
wordcloud.recolor(random_state=312)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.title("Wordcloud for countries ")
plt.axis("off")
plt.show()


# In[13]:


data_hotel_reviews.Review_Date.describe()


# In[14]:


Review_Date_count = data_hotel_reviews.Review_Date.value_counts()
plt.figure(figsize=(20,10))
Review_Date_count[:20].plot(kind='bar')


# In[15]:


the_hotels_of_countries_hist=data_hotel_reviews.groupby("hotel_country")['Reviewer_Score'].mean().reset_index().sort_values(by='hotel_country',ascending=False).reset_index(drop=True)
plt.figure(figsize=(9,6))
sns.barplot(x='hotel_country',y='Reviewer_Score', data_hotel_reviews=the_hotels_of_countries_hist)
plt.xticks(rotation=45)


# In[16]:


Reviewers_freq = data_hotel_reviews.Total_Number_of_Reviews_Reviewer_Has_Given.value_counts()
plt.figure(figsize=(20,10))
Reviewers_freq[0:10].plot(kind='bar',title='')


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


# In[68]:


min_rating = data_hotel_reviews.Average_Score.min() 
max_rating = data_hotel_reviews.Average_Score.max()
mean_rating=data_hotel_reviews.Average_Score.mean()
print('The hotel ratings are between '+ str(min_rating) +" "+'and'+" "+str(max_rating)+" "+'with a mean of'+" "+
      str(round(mean_rating,2)))


# In[69]:


data_hotel_reviews['Average_Score'].corr(data_hotel_reviews['Reviewer_Score'])


# In[56]:


correlation=data_hotel_reviews[['Additional_Number_of_Scoring','Average_Score','Review_Total_Negative_Word_Counts','Total_Number_of_Reviews','Review_Total_Positive_Word_Counts','Total_Number_of_Reviews_Reviewer_Has_Given','Reviewer_Score']]
corr_reviews=correlation.corr()
f,ax = plt.subplots(figsize=(12,9))
#Draw the heatmap using seaborn
sns.heatmap(corr_reviews, cmap='inferno', annot=True)


# In[17]:


review_data_hotel_reviews = data_hotel_reviews[['Hotel_Name', 'Positive_Review', 'Negative_Review', 'Average_Score', 'Reviewer_Score']].copy()
review_data_hotel_reviews.head()


# In[18]:


review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['Positive_Review'].astype(str) + review_data_hotel_reviews['Negative_Review'].astype(str)


# In[19]:


for i in review_data_hotel_reviews.columns:
    print(i)


# In[108]:


review_data_hotel_reviews.head()


# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')


# In[21]:


review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].fillna('')
review_data_hotel_reviews.Reviewer_Score.value_counts()
review_data_hotel_reviews.Reviewer_Score.value_counts().plot(kind='bar',figsize=(15,10))


# In[24]:


Length = len(data_hotel_reviews.Reviewer_Nationality)
result=(data_hotel_reviews.Reviewer_Nationality.value_counts()*100/Length).sort_values(ascending=False)
result
plt.xlabel('Nation')
plt.ylabel('Percentage of people visiting')
plt.title('Visitors Ratio')
result.head(10).plot(kind='pie',figsize=(20,10))
plt.show()


# In[ ]:





# In[25]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))


# In[28]:


nltk.download('stopwords')


# In[26]:


stop_words=set(stopwords.words("english"))
review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
review_data_hotel_reviews['review_text'].head()


# In[27]:


#removing english stop words
review_data_hotel_reviews['STOPWORDS'] = review_data_hotel_reviews['review_text'].apply(lambda x: len([x for x in x.split() if x in stop_words]))
print(review_data_hotel_reviews[['review_text','STOPWORDS']].head())
#removing digits
review_data_hotel_reviews['NUMERICS'] = review_data_hotel_reviews['review_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
print(review_data_hotel_reviews[['review_text','NUMERICS']].head())
#removing hashtags
review_data_hotel_reviews['HASHTAGS'] = review_data_hotel_reviews['review_text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
print(review_data_hotel_reviews[['review_text','HASHTAGS']].head())


# In[28]:


#removing punctuation
review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].str.replace('[^\w\s]','')
#trasnforming to lower case
review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[82]:





# In[83]:


review_data_hotel_reviews


# In[29]:


#commond words removal
freq = pd.Series(' '.join(review_data_hotel_reviews['review_text']).split()).value_counts()[:10]
freq


# In[ ]:





# In[30]:


#removing frequent words
freq = list(freq.index)
review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
review_data_hotel_reviews['review_text'].head()


# In[99]:


def text_freq(x)
pd.value_counts(x.split(""))

tf1 = (review_data['review_text'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']


# In[39]:


TextBlob(review_data['review_text'][0]).words


# In[89]:


frequency_text = (review_data_hotel_reviews['review_text'][1:10000]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
frequency_text.columns = ['words','frequency']
frequency_text


# In[90]:


from sklearn.model_selection import train_test_split
train,test1 = train_test_split(review_data,test_size=0.8,random_state=42)

print(train.shape);print(test1.shape)


# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer
t = TfidfVectorizer(max_features=10000)
x = t.fit_transform(review_data['review_text'].values.astype('U'))  


# In[23]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[24]:


gbdt = GradientBoostingRegressor(max_depth=5,learning_rate=0.1,n_estimators=150) # Large iteration, fewer estimators
gbdt.fit(x,review_data['review_text'])


# In[92]:


review_data_hotel_reviews.isnull().sum()


# In[91]:


review_data_hotel_reviews=review_data_hotel_reviews.dropna()
review_data_hotel_reviews = review_data_hotel_reviews.reset_index(drop=True)
review_data_hotel_reviews


# In[93]:


# 30 worst hotels
worst_hotels =data_hotel_reviews.groupby('Hotel_Name')['Reviewer_Score'].mean().sort_values(ascending=True).head(30)
worst_hotels.plot(kind="bar",color="black",figsize=(20,10))
plt.xlabel('Worst Hotels according to Reviews')
plt.ylabel('Average Review Score')
plt.rcParams.update({'font.size': 30})
plt.show()


# In[95]:


#Top 30 best hotels 
best_hotels = data_hotel_reviews.groupby('Hotel_Name')['Reviewer_Score'].mean().sort_values(ascending=False).head(30)
best_hotels.plot(kind="bar",color = "Cyan",figsize=(20,10))
plt.xlabel('Best Hotels according to Reviews')
plt.ylabel('Average Review Score')
plt.rcParams.update({'font.size': 30})
plt.show()


# In[96]:


data_hotel_reviews['Reviewer_Nationality']


# In[97]:


data_hotel_reviews['Leisure'] = data_hotel_reviews['Tags'].map(lambda x: 1 if ' Leisure trip ' in x else 0)
data_hotel_reviews['Business'] = data_hotel_reviews['Tags'].map(lambda x: 2 if ' Business trip ' in x else 0)
data_hotel_reviews['Trip_type'] = data_hotel_reviews['Leisure'] + data_hotel_reviews['Business']


# In[38]:


data_hotel_reviews['Trip_type'].value_counts()


# In[99]:


import random
data_hotel_reviews['Trip_type'] = data_hotel_reviews[data_hotel_reviews['Trip_type'] == 0]['Trip_type']


# In[101]:


data_hotel_reviews['Trip_type'] = data_hotel_reviews['Trip_type'].fillna(0)


# In[121]:



pd.set_option("max_rows", None)
data_hotel_reviews


# In[41]:


for i in copy.columns:
    print(i)


# In[119]:


import random
rand1=random.random()
def trip_type(x):
    if rand1 > 0.2:
        return 1
    else:
        return 2
data_hotel_reviews['Trip_type'] = data_hotel_reviews[data_hotel_reviews['Trip_type'] == 0]['Trip_type'].map(trip_type)
data_hotel_reviews['Trip_type'] = data_hotel_reviews['Trip_type'].fillna(0)
data_hotel_reviews['Trip_type'] = data_hotel_reviews['Trip_type'] + data_hotel_reviews['Business'] + data_hotel_reviews['Leisure']
data_hotel_reviews['Trip_type'].value_counts()


# In[43]:


copy=data_hotel_reviews


# In[45]:


Tags_data=copy.Tags
Tags_1=Tags_data.str.split("[",n=1,expand=True)
Tags=Tags_data.str.split(",", n = 4, expand = True)
Tags


# In[47]:


copy['Trip_Type']=Tags[0]
copy['Traveller_Type']=Tags[1]
copy['Room_Type']=Tags[2]
copy['Duration_of_Stay']=Tags[3]
copy['Booking_Method']=Tags[4]
trip=copy.Trip_Type
trip_data=trip.str.split("[",n=1,expand=True)
copy.Trip_Type=trip_data[1]
duration=copy.Duration_of_Stay
duration_data=duration.str.split("]",n=1,expand=True)
copy.Duration_of_Stay=duration_data[0]
method=copy.Booking_Method
booking_method=method.str.split("]",n=1,expand=True)
copy.Booking_Method=booking_method[0]


# In[ ]:


data_hotel_reviews['Solo'] = data_hotel_reviews['Tags'].map(lambda x: 1 if ' Solo traveler ' in x else 0)

data_hotel_reviews['Couple'] = data_hotel_reviews['Tags'].map(lambda x: 2 if ' Couple ' in x else 0)

data_hotel_reviews['Group'] = data_hotel_reviews['Tags'].map(lambda x: 3 if ' Group ' in x else 0)

data_hotel_reviews['Family_with_young_children'] = data_hotel_reviews['Tags'].map(lambda x: 4 if ' Family with young children ' in x else 0)

data_hotel_reviews['Family_with_older_children'] = data_hotel_reviews['Tags'].map(lambda x: 5 if ' Family with older children ' in x else 0)

data_hotel_reviews['Traveller_type'] =data_hotel_reviews['Solo'] +data_hotel_reviews['Couple'] + data_hotel_reviews['Group'] + data_hotel_reviews['Family_with_young_children'] + data_hotel_reviews['Family_with_older_children']
del data_hotel_reviews['Solo'],data_hotel_reviews['Couple'],data_hotel_reviews['Group'], data_hotel_reviews['Family_with_young_children'],data_hotel_reviews['Family_with_older_children']


# In[ ]:





# In[99]:





# In[107]:



#assigning the highest value to the 0's
data_hotel_reviews['Traveller_type']=data_hotel_reviews['Traveller_type'].map(lambda x:2 if x==0 else x)

data_hotel_reviews.Traveller_type.value_counts()


# In[65]:


copy.Booking_Method


# In[58]:


copy.Hotel_Name.unique()


# In[ ]:


featuring_negative_words


# In[32]:



nltk.download('wordnet')


# In[170]:


review_data_hotel_reviews['Sentiment_length'] = review_data_hotel_reviews.apply(lambda row: len(row['review_text']), axis=1)
review_data_hotel_reviews['Sentiment_length']


# In[49]:


review_data_hotel_reviews['Tokenized_Sentences'] =review_data_hotel_reviews["review_text"].apply(nltk.word_tokenize)


# In[171]:


review_data_hotel_reviews


# In[59]:


useless_english_words = nltk.corpus.stopwords.words("english")


# In[58]:


def filtering_bag_of_words(words):
        return{word:1 for word in words 
        if not word in useless_english_words}


# In[63]:


all_negative_review = data_hotel_reviews.Negative_Review


# In[64]:


negative_words = []
for i in range(515738):
    negative_words.append(nltk.word_tokenize(all_negative_review.iloc[i]))


# In[66]:


all_positive_review=data_hotel_reviews.Positive_Review


# In[71]:


postive_words = []
for i in range(515738):
    postive_words.append(nltk.word_tokenize(all_positive_review.iloc[i]))


# In[82]:


featuring_positive_words = None
featuring_positive_words = [(filtering_bag_of_words(review),'positive')for review in postive_words]


# In[ ]:





# In[83]:


featuring_negative_words = None
featuring_negative_words = [(filtering_bag_of_words(review),'Negative')for review in negative_words]


# In[84]:


from nltk.classify import NaiveBayesClassifier


# In[85]:


#Using 70% of the data for training, the 30% for validation:
split = int(len(featuring_positive_words) * 0.7)
split


# In[88]:


#Naive Bayes Classification
classifier = NaiveBayesClassifier.train(featuring_positive_words[:split]+featuring_negative_words[:split])


# In[89]:


training_accuracy = None #check accuracy of training set
training_accuracy = nltk.classify.util.accuracy(classifier, featuring_positive_words[:split] + featuring_negative_words[:split])*100
training_accuracy


# In[90]:


test_accuracy = None #check accuracy of test set
test_accuracy = nltk.classify.util.accuracy(classifier, featuring_positive_words[split:] +featuring_negative_words[split:])*100
test_accuracy


# In[ ]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data, stratify = data['TripType'], test_size = 0.30, random_state = 1)


# In[147]:


train,test1 = train_test_split(review_data_hotel_reviews,test_size=0.7,random_state=50)
train1,test2 = train_test_split(test1,test_size=0.8,random_state=50)
print(train.shape);print(test1.shape);print(test2.shape);print(train1.shape)


# In[148]:


from sklearn.feature_extraction.text import TfidfVectorizer
t = TfidfVectorizer(max_features=10000)
train1_features = t.fit_transform(train['review_text'])
test1_features = t.transform(test1['review_text'])
test2 = t.transform(test2['review_text'])
train2_features=t.transform(train1['review_text'])


# In[ ]:


#Gradient Boosting Regressor


# In[149]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
gbdt = GradientBoostingRegressor(max_depth=5,learning_rate=0.1,n_estimators=150)
gbdt.fit(train1_features,train['Reviewer_Score'])
gbdt.fit(train2_features,train1['Reviewer_Score'])


# In[151]:


prediction_primary = gbdt.predict(train1_features)
prediction_secondary=gbdt.predict(train2_features)


# In[158]:


MAE=mean_absolute_error(train['Reviewer_Score'],prediction_primary)
MAE_Secondary=mean_absolute_error(train1['Reviewer_Score'],prediction_secondary)


# In[159]:


RMSE=mean_squared_error(train['Reviewer_Score'],prediction_primary)
RMSE_Secondary=mean_absolute_error(train1['Reviewer_Score'],prediction_secondary)


# In[162]:


RMSE


# In[166]:


RMSE_Secondary


# In[165]:


MAE


# In[164]:


MAE_Secondary


# In[167]:


#taking a fraction of the data for faster computation
fraction_reviews = data_hotel_reviews.sample(frac = 0.1, replace = False, random_state=42)


# In[169]:


fraction_reviews.shape


# In[177]:


review_data_hotel_reviews


# In[179]:


text = ""
for i in range(review_data_hotel_reviews.shape[0]):
    text = " ".join([text,review_data_hotel_reviews["Tokenized_Sentences"].values[i]])
    

wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text)
wordcloud.recolor(random_state=312)
plt.imshow(wordcloud)
plt.title("Wordcloud for reveiws ")
plt.axis("off")
plt.show()


# In[ ]:




