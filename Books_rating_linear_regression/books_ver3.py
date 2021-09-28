import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score

# Read data

data = pd.read_csv('books.csv', error_bad_lines=False)

# Data Exploration

data.head()              
data.shape               
data.describe()          
data.info()
data.isnull().any()      
data.duplicated().any() 
sns.heatmap(data.isnull())

sns.barplot(data['average_rating'].value_counts().head(20).index, data['average_rating'].value_counts().head(20))
plt.title('Number of Books Each Rating Received\n')
plt.xlabel('Ratings')
plt.ylabel('Counts')
plt.xticks(rotation=90)

# ratings distribution
sns.kdeplot(data['average_rating'])
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Frequency')

# highest rated books
popular_books = data.nlargest(10, ['ratings_count']).set_index('title')['ratings_count']
sns.barplot(popular_books, popular_books.index)

# authors with highest rated books
plt.figure(figsize=(10, 5))
authors = data.nlargest(5, ['ratings_count']).set_index('authors')
sns.barplot(authors['ratings_count'], authors.index, ci = None, hue = authors['title'])
plt.xlabel('Total Ratings')

# top languages
data['language_code'].value_counts().plot(kind='bar')

plt.title('Most Popular Language')
plt.ylabel('Counts')
plt.xticks(rotation = 90)
data['language_code'].value_counts().head(6).plot(kind = 'pie', autopct='%1.1f%%', figsize=(9, 9)).legend()

# authors with smallets rated books
plt.figure(figsize=(10, 5))
authors = data.nsmallest(5, ['ratings_count']).set_index('authors')
sns.barplot(authors['ratings_count'], authors.index, ci = None, hue = authors['title'])
plt.xlabel('Total Ratings')

# top 10 longest books
longest_books = data.nlargest(10, ['  num_pages']).set_index('title')
sns.barplot(longest_books['  num_pages'], longest_books.index)

# authors with highest publications
top_authors = data['authors'].value_counts().head(10)
sns.barplot(top_authors, top_authors.index)
plt.title('Authors with Highest Publication Count')
plt.xlabel('No. of Publications')

# top published books
sns.barplot(data['title'].value_counts()[:15], data['title'].value_counts().index[:15])
plt.title('Top Published Books')
plt.xlabel('Number of Publications')

# visualise a bivariate distribution between ratings & no. of pages
sns.jointplot(x = 'average_rating', y = '  num_pages', data = data)

# visualise a bivariate distribution between ratings & no. of reviews
sns.jointplot(x = 'average_rating', y = 'text_reviews_count', data = data)


# Data preprocessing

# find no. of pages outliers
sns.boxplot(x=data['  num_pages'])   
np.percentile(data['  num_pages'],[99])  
np.percentile(data['  num_pages'],[99])[0] 
uv = np.percentile(data['  num_pages'],[99])[0] 
datos_mayores_que_uv =data[data['  num_pages'] > uv]
datos_mayores_que_uv   
data['  num_pages'][(data['  num_pages']> 3*uv)] = 3*uv
datos_mayores_que_uv =data[data['  num_pages'] > uv]
datos_mayores_que_uv
sns.boxplot(x=data['  num_pages'])  

 
# find ratings count outliers
sns.boxplot(x=data['ratings_count'])
np.percentile(data['ratings_count'],[99])  
np.percentile(data['ratings_count'],[99])[0] 
uw = np.percentile(data['ratings_count'],[99])[0] 
datos_mayores_que_uw =data[data['ratings_count'] > uw]
datos_mayores_que_uw   
data['ratings_count'][(data['ratings_count']> 3*uw)] = 3*uw
datos_mayores_que_uw =data[data['ratings_count'] > uw]
datos_mayores_que_uw
sns.boxplot(x=data['ratings_count']) 


# find ratings count outliers
sns.boxplot(x=data['text_reviews_count'])       
np.percentile(data['text_reviews_count'],[99])  
np.percentile(data['text_reviews_count'],[99])[0] 
ux = np.percentile(data['text_reviews_count'],[99])[0] 
datos_mayores_que_ux =data[data['text_reviews_count'] > ux]
datos_mayores_que_ux   
data['text_reviews_count'][(data['text_reviews_count']> 3*ux)] = 3*ux
datos_mayores_que_ux =data[data['text_reviews_count'] > ux]
datos_mayores_que_ux
sns.boxplot(x=data['text_reviews_count']) 



#Feature Engineering

# encode title column
le = preprocessing.LabelEncoder()
data['title'] = le.fit_transform(data['title'])

# encode authors column
data['authors'] = le.fit_transform(data['authors'])

# encode language column
enc_lang = pd.get_dummies(data['language_code'])
data = pd.concat([data, enc_lang], axis = 1)

#encode publisher
data['publisher'] = le.fit_transform(data['publisher'])

#encode publication_date
data['publication_date'] = le.fit_transform(data['publication_date'])

correlacion = data.corr()


# Machine Learning Model

# divide the data into attributes and labels
X = data.drop(['average_rating', 'language_code', 'isbn', 'isbn13'], axis = 1)

y = data['average_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'X_train: ', X_train.shape
'X_test: ',X_test.shape
'y_train: ',y_train.shape
'y_test: ',y_test.shape

lr = LinearRegression()

lr.fit(X_train, y_train)

predictions = lr.predict(X_test)

pred = pd.DataFrame({'Actual': y_test.tolist(), 'Predicted': predictions.tolist()}).head(25)

pred.head(10)

# visualise the above comparison result
pred.plot(kind='bar', figsize=(13, 7))

'MAE:', metrics.mean_absolute_error(y_test, predictions)

'MSE:', metrics.mean_squared_error(y_test, predictions)

'RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))