#!/usr/bin/env python
# coding: utf-8

# In[141]:


#Importing the python packages 
import numpy as np
import pandas as pd

#Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Modules for feature extraction supported by Machine learning and for splitting datasets into random test and train subsets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#NLP Packages
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# # Loading the dataset Hotel Reviews

# In[4]:


data_hotel_reviews = pd.read_csv("Hotel_Reviews.csv")


# # To view the Top 30 records in the dataset

# In[102]:


data_hotel_reviews.head(30)


# # To get the column names of the of the dataset

# In[103]:


for col in data_hotel_reviews.columns:
    print(col)


# # Removing the duplicate values from the dataset

# In[125]:


print(sum(data_hotel_reviews.duplicated()))


# In[128]:


data_hotel_reviews=data_hotel_reviews.drop_duplicates()
data_hotel_reviews.shape


# # Exploratory Data Analysis

# # To get the unique hotel names from the dataset

# In[6]:


len(data_hotel_reviews.Hotel_Name.unique())


# # To view some basic statistical details of the dataset

# In[7]:


data_hotel_reviews.Reviewer_Nationality.describe()


# # To view the Top 30 reviewer nationalities 

# In[105]:


Reviewer_Nat_Count = data_hotel_reviews.Reviewer_Nationality.value_counts()
print(Reviewer_Nat_Count[:30])


# In[129]:


data_hotel_reviews.Hotel_Name.describe()


# # Plot of the Count of the reviews vs the Average review score 

# In[131]:


data_hotel_reviews_plot = data_hotel_reviews[["Hotel_Name","Average_Score"]].drop_duplicates()
sns.set(font_scale = 2.6) # setting the font size 
a4_dims = (30, 12)# plotting in an A4 size paper with length 22 and width 16
fig, ax = pyplot.subplots(figsize=a4_dims)
plt.xticks(rotation=50)
sns.countplot(ax = ax,x = "Average_Score",data=data_hotel_reviews)


# # To get the nationality of the reviewer and ploting a wordcloud of the nationality

# In[122]:


text = " "
for i in range(data_hotel_reviews.shape[0]):
    text = " ".join([text,data_hotel_reviews["Reviewer_Nationality"].values[i]])


# In[138]:


from wordcloud import WordCloud
wordcloud = WordCloud(background_color='black', width = 500,                      height=200, max_font_size=50, max_words=50).generate(text)
wordcloud.recolor(random_state=312)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud,interpolation='lanczos')
plt.title("Wordcloud of the reviewer countries ")
plt.axis("off")
plt.show()


# # Statistics of the Review Date column 

# In[13]:


data_hotel_reviews.Review_Date.describe()


# # Plot of the count of the reviews vs the dates for Top 20 highest review received dates

# In[14]:


Review_Date_count = data_hotel_reviews.Review_Date.value_counts()
plt.figure(figsize=(20,10))
Review_Date_count[:20].plot(kind='bar')


# In[ ]:


# Total number of countries which have given reviews


# In[155]:


countries_reviews = data_hotel_reviews.Reviewer_Nationality.unique()
distinctCountries_reviews = len(countries_reviews)
print("Total number of countries from which reviews have been made :", distinctCountries_reviews)


# In[27]:


Length = len(data_hotel_reviews.Reviewer_Nationality)
percentage_of_users_visiting= (data_hotel_reviews.Reviewer_Nationality.value_counts()*100/Length).sort_values(ascending=False)
percentage_of_users_visiting


# In[28]:


from matplotlib import pyplot as plt
plt.xlabel('Countries')
plt.ylabel('Percentage of people visiting Europe Hotels')
plt.title('Visitors Ratio')
percentage_of_users_visiting.head(20).plot(kind='bar',figsize=(20,10))
plt.show()


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


# In[10]:


review_data_hotel_reviews = data_hotel_reviews[['Hotel_Name', 'Positive_Review', 'Negative_Review', 'Average_Score', 'Reviewer_Score']].copy()
review_data_hotel_reviews.head()


# In[11]:


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


# In[13]:


stop_words=set(stopwords.words("english"))
review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
review_data_hotel_reviews['review_text'].head()


# In[14]:


#removing english stop words
review_data_hotel_reviews['STOPWORDS'] = review_data_hotel_reviews['review_text'].apply(lambda x: len([x for x in x.split() if x in stop_words]))
print(review_data_hotel_reviews[['review_text','STOPWORDS']].head())
#removing digits
review_data_hotel_reviews['NUMERICS'] = review_data_hotel_reviews['review_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
print(review_data_hotel_reviews[['review_text','NUMERICS']].head())
#removing hashtags
review_data_hotel_reviews['HASHTAGS'] = review_data_hotel_reviews['review_text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
print(review_data_hotel_reviews[['review_text','HASHTAGS']].head())


# In[15]:


#removing punctuation
review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].str.replace('[^\w\s]','')
#trasnforming to lower case
review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[82]:


res = "".join(filter(lambda x: not x.isdigit(), ini_string))


# In[16]:


review_data_hotel_reviews


# In[17]:


#commond words removal
freq = pd.Series(' '.join(review_data_hotel_reviews['review_text']).split()).value_counts()[:10]
freq


# In[ ]:





# In[18]:


#removing frequent words
freq = list(freq.index)
review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
review_data_hotel_reviews['review_text'].head()


# In[19]:


def text_freq(x)
pd.value_counts(x.split(""))

tf1 = (review_data['review_text'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']


# In[20]:


TextBlob(review_data['review_text'][0]).words


# In[21]:


frequency_text = (review_data_hotel_reviews['review_text'][1:10000]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
frequency_text.columns = ['words','frequency']
frequency_text


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


# In[83]:


TOP_hotel_Names = (data_hotel_reviews.groupby('Hotel_Name')['Reviewer_Score'].sum()/data_hotel_reviews.groupby('Hotel_Name')['Reviewer_Score'].count()).sort_values(ascending=False).head(10)
TOP_hotel_Names


# In[95]:


#Top 30 best hotels 
best_hotels = data_hotel_reviews.groupby('Hotel_Name')['Reviewer_Score'].mean().sort_values(ascending=False).head(30)
best_hotels.plot(kind="bar",color = "Cyan",figsize=(20,10))
plt.xlabel('Best Hotels according to Reviews')
plt.ylabel('Average Review Score')
plt.rcParams.update({'font.size': 30})
plt.show()


# In[58]:


TOP_hotel_Names


# In[60]:


TOP_3_hotel_names_df = ['Ritz Paris','Hotel Casa Camper','41']


# In[62]:


for hotelnames in TOP_3_hotel_names_df:
    hotelreviewintime = data_hotel_reviews[data_hotel_reviews.Hotel_Name == hotelnames]
    hotelreviewintime.plot('Review_Date', 'Reviewer_Score',figsize=(10,5))
    plt.xlabel('Review Date', fontsize=12)
    plt.ylabel('Review score', fontsize=12)
    title_str = hotelnames
    plt.title(title_str, fontsize=15)
    plt.show()
    (:,2)


# In[96]:


data_hotel_reviews['Reviewer_Nationality']


# In[39]:


data_hotel_reviews.Hotel_Name.


# In[41]:



Number_of_Leisure_Hotels = data_hotel_reviews[data_hotel_reviews['Tags'].str.contains('Leisure')].groupby('Hotel_Name')['Hotel_Name'].count().sort_values(ascending=False)
print ("Leisure Hotels: ", len(Number_of_Leisure_Hotels))

Number_of_Business_Hotels = data_hotel_reviews[data_hotel_reviews['Tags'].str.contains('Business')].groupby('Hotel_Name')['Hotel_Name'].count().sort_values(ascending=False)
print ("Business Hotels: ", len(Number_of_Business_Hotels))


# In[56]:



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


# In[46]:


for i in copy.columns:
    print(i)


# In[48]:


Tags_data=copy.Tags
Tags_1=Tags_data.str.split("[",n=1,expand=True)
Tags=Tags_data.str.split(",", n = 4, expand = True)
Tags


# In[49]:


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


# In[50]:


data_hotel_reviews['Solo'] = data_hotel_reviews['Tags'].map(lambda x: 1 if ' Solo traveler ' in x else 0)

data_hotel_reviews['Couple'] = data_hotel_reviews['Tags'].map(lambda x: 2 if ' Couple ' in x else 0)

data_hotel_reviews['Group'] = data_hotel_reviews['Tags'].map(lambda x: 3 if ' Group ' in x else 0)

data_hotel_reviews['Family_with_young_children'] = data_hotel_reviews['Tags'].map(lambda x: 4 if ' Family with young children ' in x else 0)

data_hotel_reviews['Family_with_older_children'] = data_hotel_reviews['Tags'].map(lambda x: 5 if ' Family with older children ' in x else 0)

data_hotel_reviews['Traveller_type'] =data_hotel_reviews['Solo'] +data_hotel_reviews['Couple'] + data_hotel_reviews['Group'] + data_hotel_reviews['Family_with_young_children'] + data_hotel_reviews['Family_with_older_children']
del data_hotel_reviews['Solo'],data_hotel_reviews['Couple'],data_hotel_reviews['Group'], data_hotel_reviews['Family_with_young_children'],data_hotel_reviews['Family_with_older_children']


# In[51]:



#assigning the highest value to the 0's
data_hotel_reviews['Traveller_type']=data_hotel_reviews['Traveller_type'].map(lambda x:2 if x==0 else x)

data_hotel_reviews.Traveller_type.value_counts()


# In[52]:


copy.Booking_Method


# In[32]:



nltk.download('wordnet')


# In[54]:


review_data_hotel_reviews['Sentiment_length'] = review_data_hotel_reviews.apply(lambda row: len(row['review_text']), axis=1)
review_data_hotel_reviews['Sentiment_length']


# In[55]:


review_data_hotel_reviews['Tokenized_Sentences'] =review_data_hotel_reviews["review_text"].apply(nltk.word_tokenize)


# In[12]:


review_data_hotel_reviews


# In[66]:


english_stop_words = nltk.corpus.stopwords.words("english")


# In[68]:


def filtering_bag_of_words(words):
        return{word:1 for word in words 
        if not word in english_stop_words}


# In[70]:


all_negative_review = data_hotel_reviews.Negative_Review


# In[71]:


negative_words = []
for i in range(515738):
    negative_words.append(nltk.word_tokenize(all_negative_review.iloc[i]))


# In[72]:


all_positive_review=data_hotel_reviews.Positive_Review


# In[73]:


postive_words = []
for i in range(515738):
    postive_words.append(nltk.word_tokenize(all_positive_review.iloc[i]))


# In[74]:


featuring_positive_words = None
featuring_positive_words = [(filtering_bag_of_words(review),'positive')for review in postive_words]


# In[75]:


featuring_negative_words = None
featuring_negative_words = [(filtering_bag_of_words(review),'Negative')for review in negative_words]


# In[76]:


from nltk.classify import NaiveBayesClassifier


# In[77]:


#Using 70% of the data for training, the 30% for validation:
split = int(len(featuring_positive_words) * 0.7)
split


# In[78]:


#Naive Bayes Classification
classifier = NaiveBayesClassifier.train(featuring_positive_words[:split]+featuring_negative_words[:split])


# In[80]:


training_accuracy = None #check accuracy of training set
training_accuracy = nltk.classify.util.accuracy(classifier, featuring_positive_words[:split] + featuring_negative_words[:split])*100
training_accuracy


# In[81]:


test_accuracy = None #check accuracy of test set
test_accuracy = nltk.classify.util.accuracy(classifier, featuring_positive_words[split:] +featuring_negative_words[split:])*100
test_accuracy


# In[84]:


classifier.show_most_informative_features(40)


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


#Gradient Boosting Regression model


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


# In[154]:


positive_reviews = review_data_hotel_reviews['Positive_Review'].values
positive_reviews = positive_reviews.tolist()
negative_reviews = review_data_hotel_reviews['Negative_Review'].values
negative_reviews = negative_reviews.tolist()
score = ['positive' for i in range(len(positive_reviews))]
score += ['negative' for i in range(len(negative_reviews))]
for i in range(0,len(score)):
    if score[i] == 'positive':
        score[i] = 1
    else:
        score[i] = 0


# In[86]:


sns.countplot(data=df, x='score')


# In[ ]:





# In[87]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




