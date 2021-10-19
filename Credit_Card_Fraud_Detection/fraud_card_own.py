#import libraries

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Load data
data = pd.read_csv('creditcard.csv', header=0)

# Data preparation

data.shape
data.columns 

data.head()
data.sample(5)
data.tail()

data.describe()
data.info()

data.isnull().any()
data.isnull().sum()
data.isnull().any().any()
sns.heatmap(data.isnull())



### exploratory analisis

data.hist(figsize=(40,40))
plt.show()


plt.figure(figsize=(5,4))
plt.hist(data['Class'])
plt.ylabel('Class')

sns.countplot('Class',data=data)
plt.title("class distribution")
plt.show()

## para ver el porcentaje de fraudes y transacciones reales
data.Class.value_counts()
data.Class.value_counts(normalize=True)


# Time

data['Time'].describe()
#convert time seconds to hours
data.loc[:,'Time'] = data.Time /3600
data['Time'].describe()

data['Time'].max()
data['Time'].max() /24
#transactions ocurred over 2 days 


plt.figure(figsize=(15,8))
plt.hist(data['Time'], bins=100)
plt.xlim([0,48])
plt.xlabel('Time Transaction (hr)')
plt.ylabel('Count')
plt.title('Transaction Times')


plt.figure(figsize=(15,8))
sns.jointplot(x='Time', y = 'Class', data= data, kind='hex')
plt.title('Fraud and real Transaction Times')

### Amount
data['Amount'].describe()

plt.figure(figsize=(15,8))
plt.hist(data['Amount'], bins=300)
plt.ylabel('Count')
plt.title('Transaction Amounts')

plt.figure(figsize=(15,8))
plt.hist(data['Amount'], bins=1000)
plt.xlim([0,1500])
plt.ylabel('Count')
plt.title('Transaction Amounts')

plt.figure(figsize=(15,8))
sns.jointplot(x='Amount', y = 'Class', data= data, kind='hex')
plt.xlim([0,1500])
plt.title('Fraud and real Transaction Times')


# outlier values
plt.figure(figsize=(15,8))
sns.boxplot(x=data['Amount'])  
plt.title('Transaction Amounts')


np.percentile(data.Amount,[99])
np.percentile(data.Amount,[99])[0]
uv = np.percentile(data.Amount,[99])[0]
data = data.drop(data.index[data['Amount'] >= uv])

plt.figure(figsize=(15,8))
sns.boxplot(x=data['Amount'])  
plt.title('Transaction Amounts')

data.shape

data['Amount'].describe()

# Time vs. Amount

plt.figure(figsize=(15,8))
sns.jointplot(x='Time', y = 'Amount', data= data, kind='hex',bins=100)
plt.title('Transaction Amounts')

plt.figure(figsize=(15,8))
sns.jointplot(x='Time', y = 'Amount', data= data, kind='hex',bins=100)
plt.ylim([0, 300])
plt.title('Transaction Amounts')


# Scatter plot of Class vs Amount and Time for Normal Transactions 

plt.figure(figsize=(15,8))
fig = plt.scatter(x=data[data['Class'] == 0]['Time'], y=data[data['Class'] == 0]['Amount'])
plt.title("Time vs Transaction Amount in Normal Transactions")
plt.xlabel("Time (in hours)")
plt.ylabel("Amount of Transaction")


plt.figure(figsize=(15,8))
fig = plt.scatter(x=data[data['Class'] == 1]['Time'], y=data[data['Class'] == 1]['Amount'])
plt.title("Time vs Transaction Amount in Fraud Cases")
plt.xlabel("Time (in hours)")
plt.ylabel("Amount of Transaction")
#there are much more outliers as compared to normal transactions

#### V1-V28

pca_vars = ['V%i' % k for k in range(1,29)]
v1_v28 = data[pca_vars].describe()
data[pca_vars].describe()

## graficamos las media para un analisi mas senillo
vs_mean =  data[pca_vars].mean()

plt.figure(figsize=(15,5))
sns.barplot(x =pca_vars, y =vs_mean)
plt.xlabel('Column')
plt.ylabel('Mean')
plt.title('V1-V28 Means')

data[pca_vars].mean().mean()

#All of V1-V28 have approximately zero mean. Now plot the standard deviations:
    
plt.figure(figsize=(15,5))
sns.barplot(x=pca_vars, y=data[pca_vars].std())
plt.xlabel('Column')
plt.ylabel('Standard Deviation')
plt.title('V1-V28 Standard Deviations')    

data[pca_vars].std().max()

data[pca_vars].std().min()

# The PCA variables have roughly unit variance, but as low as ~0.3 and as high as ~1.9.
#  Let's plot the medians:
    
plt.figure(figsize=(15,5))
sns.barplot(x=pca_vars, y=data[pca_vars].median())
plt.xlabel('Column')
plt.ylabel('Median')
plt.title('V1-V28 Medians')    
    
data[pca_vars].median().mean()
#The medians are also roughly zero. Next let's look at the interquartile ranges (IQR)*:

plt.figure(figsize=(15,5))
sns.barplot(x=pca_vars,
            y=data[pca_vars].quantile(0.75) - data[pca_vars].quantile(0.25))
plt.xlabel('Column')
plt.ylabel('IQR')
plt.title('V1-V28 IQRs')


correlacion = data.corr()
sns.heatmap(data.corr())


from sklearn.model_selection import train_test_split

X= data.drop('Class',axis=1)
X.head()

y = data['Class']
y.head()

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=0)

X_train.shape
X_test.shape
y_train.shape
y_test.shape


import statsmodels.api as sn
from sklearn.linear_model import LinearRegression



X_cons = sn.add_constant(X)
X_cons.head()

lm = sn.OLS(y, X_cons).fit()
lm.summary()

lm_a =LinearRegression()
lm_a.fit(X_train, y_train)

y_test_a = lm_a.predict(X_test)

y_train_a = lm_a.predict(X_train)

from sklearn.metrics import r2_score

r2_score(y_test, y_test_a)
r2_score(y_train, y_train_a)


# Trainning classification tree
from sklearn import tree

clftree = tree.DecisionTreeClassifier(max_depth = 3)
clftree.fit(X_train, y_train)

y_train_pred = clftree.predict(X_train) 
y_test_pred = clftree.predict(X_test)


### model performance
from sklearn.metrics import accuracy_score, confusion_matrix

cmy_train = confusion_matrix(y_train, y_train_pred)

conf_matrix=pd.DataFrame(data=cmy_train,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")

cmy_test =confusion_matrix(y_test, y_test_pred)

conf_matrix=pd.DataFrame(data=cmy_test,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")

accuracy_score(y_test, y_test_pred)
## 0.9992339338913321


# plotting decission tree
dot_data = tree.export_graphviz(clftree, out_file=None, feature_names=X_train.columns, filled=True)
from IPython.display import Image
import pydotplus
graph =pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


#controlling tree growth 

clftree2 = tree.DecisionTreeClassifier(min_samples_leaf = 20, max_depth=4)

clftree2.fit(X_train , y_train)

dot_data = tree.export_graphviz(clftree2, out_file=None, feature_names=X_train.columns, filled=True)

graph2 =pydotplus.graph_from_dot_data(dot_data)

Image(graph2.create_png())

accuracy_score(y_test, clftree2.predict(X_test))
# 0.9993332387572705

# Random forest 
from sklearn.ensemble import RandomForestClassifier       

rf_clf = RandomForestClassifier(n_estimators = 50, n_jobs = -1, random_state = 1)                 

rf_clf.fit(X_train, y_train)

rf_clf.predict(X_train)

cm_rf =confusion_matrix(y_test, rf_clf.predict(X_test))

conf_matrix=pd.DataFrame(data=cm_rf,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")

accuracy_score(y_test, rf_clf.predict(X_test))
# 0.9994751028514683

# Grid search
from sklearn.model_selection import GridSearchCV    

rf_clf = RandomForestClassifier(n_estimators = 20, random_state = 1)

params_grid = {'max_features': [3,4,5],
               'min_samples_split': [2,3,5]}

grid_search = GridSearchCV(rf_clf, params_grid,
                           n_jobs=-1, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

grid_search.best_params_

cvrf_clf = grid_search.best_estimator_
cvrf_clf

accuracy_score(y_test, cvrf_clf.predict(X_test))
# 0.9995034756703078

cm_brf = confusion_matrix(y_test, cvrf_clf.predict(X_test))

conf_matrix=pd.DataFrame(data=cm_brf ,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")

