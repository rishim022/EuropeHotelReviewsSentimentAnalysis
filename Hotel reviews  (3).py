#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer

#Machine Learning packages
from nltk.classify import NaiveBayesClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# # Loading the dataset Hotel Reviews

# In[2]:


data_hotel_reviews = pd.read_csv("Hotel_Reviews.csv")


# # To view the Top 30 records in the dataset

# In[3]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data_hotel_reviews.head(30)


# # To get the column names of the of the dataset

# In[3]:


for col in data_hotel_reviews.columns:
    print(col)


# # Removing the duplicate values from the dataset

# In[125]:


print(sum(data_hotel_reviews.duplicated()))


# In[85]:


data_hotel_reviews.isnull().sum()


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


# In[99]:



Number_of_Leisure_Hotels = data_hotel_reviews[data_hotel_reviews['Tags'].str.contains('Leisure')].groupby('Hotel_Name')['Hotel_Name'].count().sort_values(ascending=False)
print ("Leisure Hotels: ", len(Number_of_Leisure_Hotels))

Number_of_Business_Hotels = data_hotel_reviews[data_hotel_reviews['Tags'].str.contains('Business')].groupby('Hotel_Name')['Hotel_Name'].count().sort_values(ascending=False)
print ("Business Hotels: ", len(Number_of_Business_Hotels))


# # Plot of the Count of the reviews vs the Average review score 

# In[7]:


data_hotel_reviews_plot = data_hotel_reviews[["Hotel_Name","Average_Score"]].drop_duplicates() 
sns.set(font_scale=2)
a4_dims = (30, 12)
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
plt.imshow(wordcloud,interpolation='lanczos')#interpolation of the image lanczos for a better quality image
plt.title("Wordcloud of the reviewer countries ")
plt.axis("off")
plt.show()


# # Statistics of the Review Date column 

# In[13]:


data_hotel_reviews.Review_Date.describe()


# # Plot of the count of the reviews vs the dates for Top 20 highest review received dates

# In[41]:


Review_Date_count = data_hotel_reviews.Review_Date.value_counts()
plt.figure(figsize=(20,10))
plt.title("Bar chart of the count of reviews vs the dates for Top 20 highest review received dates ")
plt.xlabel("Date")
plt.ylabel("Reviews given")
Review_Date_count[:20].plot(kind='bar')


# # Total number of countries which have given reviews

# In[155]:


countries_reviews = data_hotel_reviews.Reviewer_Nationality.unique()
distinctCountries_reviews = len(countries_reviews)
print("Total number of countries from which reviews have been made :", distinctCountries_reviews)


# # Top reviewer countries and the percentage of contribution to the dataset

# In[16]:


Length = len(data_hotel_reviews.Reviewer_Nationality)
percentage_of_users_visiting= (data_hotel_reviews.Reviewer_Nationality.value_counts()*100/Length).sort_values(ascending=False)
percentage_of_users_visiting


# # Visitor ratio(Percentage of people visiting Europe's Hotels vs the Countries)

# In[40]:


from matplotlib import pyplot as plt
plt.xlabel('Countries')
plt.ylabel('Percentage of people visiting Europe Hotels')
plt.title('Visitors Ratio')
percentage_of_users_visiting.head(20).plot(kind='bar',figsize=(20,10))
plt.show()


# # Count of the reviews given vs Total number of reviews user has given 

# In[29]:


Reviewers_freq = data_hotel_reviews.Total_Number_of_Reviews_Reviewer_Has_Given.value_counts()
plt.figure(figsize=(20,10))
plt.title("Bar plot of the Frequency of the number of reviews vs the Count of the number of reviews")
plt.xlabel("Count of the number of reviews given by a reviewer")
plt.ylabel("Frequency of the number of reviews given")
Reviewers_freq[0:20].plot(kind='bar')
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 21}
plt.rc('font', **font)


# # Minimum, maximum and the mean ratings givne to the Hotel

# In[68]:


min_rating = data_hotel_reviews.Average_Score.min() 
max_rating = data_hotel_reviews.Average_Score.max()
mean_rating=data_hotel_reviews.Average_Score.mean()
print('The hotel ratings are between '+ str(min_rating) +" "+'and'+" "+str(max_rating)+" "+'with a mean of'+" "+
      str(round(mean_rating,2)))


# In[69]:


data_hotel_reviews['Average_Score'].corr(data_hotel_reviews['Reviewer_Score'])


# # Correlation plot of all the columns in the dataset

# In[21]:


correlation=data_hotel_reviews[['Additional_Number_of_Scoring','Average_Score','Review_Total_Negative_Word_Counts','Total_Number_of_Reviews','Review_Total_Positive_Word_Counts','Total_Number_of_Reviews_Reviewer_Has_Given','Reviewer_Score']]
corr_reviews=correlation.corr()
f,ax = plt.subplots(figsize=(15,12))
sns.heatmap(corr_reviews, cmap='inferno', annot=True)


# # Extraction of specific columns from the dataset

# In[22]:


review_data_hotel_reviews = data_hotel_reviews[['Hotel_Name', 'Positive_Review', 'Negative_Review', 'Average_Score', 'Reviewer_Score']].copy()
review_data_hotel_reviews.head()


# # Concatination of postive and negative reviews

# In[23]:


review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['Positive_Review'].astype(str) + review_data_hotel_reviews['Negative_Review'].astype(str)


# In[19]:


for i in review_data_hotel_reviews.columns:
    print(i)


# In[108]:


review_data_hotel_reviews.head()


# # Bar plot of the number of reviews vs the reviewer score

# In[24]:


review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].fillna('')
review_data_hotel_reviews.Reviewer_Score.value_counts()
plt.title("Bar plot of the number of reviews vs the reviewer score")
plt.xlabel("Reviewer Score")
plt.ylabel("Number of reviews given")
review_data_hotel_reviews.Reviewer_Score.value_counts().plot(kind='bar',figsize=(15,10))


# # Pie Chart of the Visitor Ratio

# In[24]:


Length = len(data_hotel_reviews.Reviewer_Nationality)
result=(data_hotel_reviews.Reviewer_Nationality.value_counts()*100/Length).sort_values(ascending=False)
result
plt.xlabel('Nation')
plt.ylabel('Percentage of people visiting')
plt.title('Visitors Ratio')
result.head(10).plot(kind='pie',figsize=(20,10))
plt.show()


# # 30 worst hotels

# In[89]:


TOP_hotel_Names = (data_hotel_reviews.groupby('Hotel_Name')['Reviewer_Score'].sum()/data_hotel_reviews.groupby('Hotel_Name')['Reviewer_Score'].count()).sort_values(ascending=True).head(10)
TOP_hotel_Names


# In[25]:



worst_hotels =data_hotel_reviews.groupby('Hotel_Name')['Reviewer_Score'].mean().sort_values(ascending=True).head(30)
worst_hotels.plot(kind="bar",color="black",figsize=(20,10))
plt.xlabel('Worst Hotels according to Reviews')
plt.ylabel('Average Review Score')
plt.rcParams.update({'font.size': 30})
plt.show()


# In[88]:


TOP_hotel_Names = (data_hotel_reviews.groupby('Hotel_Name')['Reviewer_Score'].sum()/data_hotel_reviews.groupby('Hotel_Name')['Reviewer_Score'].count()).sort_values(ascending=False).head(10)
TOP_hotel_Names


# # Top 30 best hotels

# In[26]:



best_hotels = data_hotel_reviews.groupby('Hotel_Name')['Reviewer_Score'].mean().sort_values(ascending=False).head(30)
best_hotels.plot(kind="bar",color = "Cyan",figsize=(20,10))
plt.xlabel('Best Hotels according to Reviews')
plt.ylabel('Average Review Score')
plt.rcParams.update({'font.size': 30})
plt.show()


# In[27]:


TOP_3_hotel_names_df = ['Ritz Paris','Hotel Casa Camper','41']


# In[31]:


for hotelnames in TOP_3_hotel_names_df:
    hotelreviewtime = data_hotel_reviews[data_hotel_reviews.Hotel_Name == hotelnames]
    hotelreviewtime.plot('Review_Date', 'Reviewer_Score',figsize=(10,5))
    plt.xlabel('Review Date', fontsize=12)
    plt.ylabel('Review score', fontsize=12)
    title_str = hotelnames
    plt.xticks(rotation=45)
    plt.title(title_str, fontsize=15)
    plt.show()


# # Pre-processsing of the dataset

# In[112]:


review_data_hotel_reviews


# # Downloading the english stopwords

# In[ ]:


nltk.download('stopwords')


# # Using lambda expressions to removing english stop words and the special characters in the sentence 

# In[ ]:


# removing english stop words
review_data_hotel_reviews['STOPWORDS'] = review_data_hotel_reviews['review_text'].apply(lambda x: len([x for x in x.split() if x in stop_words]))
print(review_data_hotel_reviews[['review_text','STOPWORDS']].head())
# removing digits
review_data_hotel_reviews['NUMERICS'] = review_data_hotel_reviews['review_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
print(review_data_hotel_reviews[['review_text','NUMERICS']].head())
# removing hashtags
review_data_hotel_reviews['HASHTAGS'] = review_data_hotel_reviews['review_text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
print(review_data_hotel_reviews[['review_text','HASHTAGS']].head())


# In[15]:


#removing punctuation
review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].str.replace('[^\w\s]','')
#trasnforming to lower case
review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# # Removing frequent words

# In[ ]:


freq = pd.Series(' '.join(review_data_hotel_reviews['review_text']).split()).value_counts()[:10]
freq


# In[ ]:


freq = list(freq.index)
review_data_hotel_reviews['review_text'] = review_data_hotel_reviews['review_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
review_data_hotel_reviews['review_text'].head()


# # Frequency of words

# In[ ]:


frequency_text = (review_data_hotel_reviews['review_text'][1:10000]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
frequency_text.columns = ['words','frequency']
frequency_text


# # Stemming to remove suffices like "ing","ly","s"

# In[120]:


st = PorterStemmer()
review_data_hotel_reviews['review_text'][:10].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# # Creation of N-grams

# In[94]:


from textblob import TextBlob
TextBlob(review_data_hotel_reviews['review_text'][0]).ngrams(10)


# # Cloning the dataframe

# In[101]:


copy=data_hotel_reviews


# In[102]:


for i in copy.columns:
    print(i)


# # Splitting the Tags columnm

# In[103]:


Tags_data=copy.Tags
Tags_1=Tags_data.str.split("[",n=1,expand=True)
Tags=Tags_data.str.split(",", n = 4, expand = True)
Tags


# In[105]:


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


# # assigning values to the various types of traveller 

# In[106]:


data_hotel_reviews['Solo'] = data_hotel_reviews['Tags'].map(lambda x: 1 if ' Solo traveler ' in x else 0)
data_hotel_reviews['Couple'] = data_hotel_reviews['Tags'].map(lambda x: 2 if ' Couple ' in x else 0)

data_hotel_reviews['Group'] = data_hotel_reviews['Tags'].map(lambda x: 3 if ' Group ' in x else 0)

data_hotel_reviews['Family_with_young_children'] = data_hotel_reviews['Tags'].map(lambda x: 4 if ' Family with young children ' in x else 0)

data_hotel_reviews['Family_with_older_children'] = data_hotel_reviews['Tags'].map(lambda x: 5 if ' Family with older children ' in x else 0)

data_hotel_reviews['Traveller_type'] =data_hotel_reviews['Solo'] +data_hotel_reviews['Couple'] + data_hotel_reviews['Group'] + data_hotel_reviews['Family_with_young_children'] + data_hotel_reviews['Family_with_older_children']
del data_hotel_reviews['Solo'],data_hotel_reviews['Couple'],data_hotel_reviews['Group'], data_hotel_reviews['Family_with_young_children'],data_hotel_reviews['Family_with_older_children']


# # assigning the highest value to the 0's

# In[108]:


data_hotel_reviews['Traveller_type']=data_hotel_reviews['Traveller_type'].map(lambda x:2 if x==0 else x)
data_hotel_reviews.Traveller_type.value_counts()


# In[114]:


copy.head(2)


# In[32]:



nltk.download('wordnet')


# In[115]:


review_data_hotel_reviews['Sentiment_length'] = review_data_hotel_reviews.apply(lambda row: len(row['review_text']), axis=1)
review_data_hotel_reviews['Sentiment_length']


# # Tokenize the sentences in the column 

# In[116]:


review_data_hotel_reviews['Tokenized_Sentences'] =review_data_hotel_reviews["review_text"].apply(nltk.word_tokenize)


# In[12]:


review_data_hotel_reviews


# In[121]:



english_stop_words = nltk.corpus.stopwords.words("english")


# # Function to filter out the english stop words

# In[128]:


def filtering_bag_of_words(words):
        return{word:1 for word in words 
        if not word in english_stop_words}


# # Fetch all the negative reviews in a variable

# In[123]:


all_negative_review = data_hotel_reviews.Negative_Review


# # Tokenize all negative reviews into words from a column and store it in a variable

# In[124]:


negative_words = []
for i in range(515738):
    negative_words.append(nltk.word_tokenize(all_negative_review.iloc[i]))


# # Fetch all the positive reviews in a variable

# In[72]:


all_positive_review=data_hotel_reviews.Positive_Review


# # Tokenize all positive reviews into words from a column and store it in a variable

# In[73]:


postive_words = []
for i in range(515738):
    postive_words.append(nltk.word_tokenize(all_positive_review.iloc[i]))


# # Filtering all the english stop words from the positive and negative tokenized words and assigning the rest with a tag

# In[129]:


featuring_positive_words = None
featuring_positive_words = [(filtering_bag_of_words(review),'positive')for review in postive_words]


# In[130]:


featuring_negative_words = None
featuring_negative_words = [(filtering_bag_of_words(review),'Negative')for review in negative_words]


# In[76]:


from nltk.classify import NaiveBayesClassifier


# # Taking 70% of the data for training, the 30% for validation

# In[77]:



split = int(len(featuring_positive_words) * 0.7)
split


# # Naive Bayes Classification for sentiment analysis

# In[78]:



classifier = NaiveBayesClassifier.train(featuring_positive_words[:split]+featuring_negative_words[:split])


# # To check the accuracy of training set

# In[80]:


training_accuracy = None 
training_accuracy = nltk.classify.util.accuracy(classifier, featuring_positive_words[:split] + featuring_negative_words[:split])*100
training_accuracy


# # To check the accuracy of test set

# In[81]:


test_accuracy = None 
test_accuracy = nltk.classify.util.accuracy(classifier, featuring_positive_words[split:] +featuring_negative_words[split:])*100
test_accuracy


# # Classification of the words on the basis of positive and negative words

# In[84]:


classifier.show_most_informative_features(40)


# In[34]:


train,test1 = train_test_split(review_data_hotel_reviews,test_size=0.7,random_state=50)
train1,test2 = train_test_split(test1,test_size=0.8,random_state=50)
print(train.shape);
print(test1.shape);
print(test2.shape);
print(train1.shape)


# In[35]:


from sklearn.feature_extraction.text import TfidfVectorizer
t = TfidfVectorizer(max_features=10000)
train1_features = t.fit_transform(train['review_text'])
test1_features = t.transform(test1['review_text'])
test2 = t.transform(test2['review_text'])
train2_features=t.transform(train1['review_text'])


# # Gradient Boosting Classification Model

# In[32]:


gbdt = GradientBoostingRegressor(max_depth=5,learning_rate=0.1,n_estimators=150)
gbdt.fit(train1_features,train['Reviewer_Score'])
gbdt.fit(train2_features,train1['Reviewer_Score'])


# In[151]:


prediction_primary = gbdt.predict(train1_features)
prediction_secondary=gbdt.predict(train2_features)


# # Mean absolute error

# In[158]:


MAE=mean_absolute_error(train['Reviewer_Score'],prediction_primary)
MAE_Secondary=mean_absolute_error(train1['Reviewer_Score'],prediction_secondary)


# # Root Mean square error

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


# In[160]:


gbdt.plot_importance(bst)


# In[ ]:





# In[ ]:




