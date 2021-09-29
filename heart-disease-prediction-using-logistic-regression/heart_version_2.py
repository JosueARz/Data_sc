import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
%matplotlib inline
import statsmodels.api as sm
from statsmodels.tools import add_constant 
import scipy.stats as st
from sklearn.metrics import confusion_matrix

sns.set(style = 'darkgrid')
sns.set_palette('deep')

data=pd.read_csv('framingham.csv', header=0)

data.shape

data.describe()

data.info()

del data['education']

#missing values
data.isnull().any()        

data.isnull().sum()

# cigsPerDay
# BPMeds
# totChol 
# BMI                 
# heartRate           
# glucose        9.155261915998113 %



data.isnull().sum().sum()    #  0.008494572911750826 missing values percent

sns.heatmap(data.isnull())

# duplicated values
data.duplicated().any() 

data.dropna(axis=0,inplace=True)

data.isnull().sum()

sns.heatmap(data.isnull())

# Exploratory Analysis

plt.hist(data['male'])
plt.title('Sex distribution')
plt.xlabel('Sex')

plt.hist(data['age'])
plt.title('Age distribution')
plt.xlabel('Age')

plt.hist(data['currentSmoker'])
plt.title('currentSmoker distribution')
plt.xlabel('currentSmoker')

plt.hist(data['cigsPerDay'])
plt.title('cigsPerDay distribution')
plt.xlabel('cigsPerDay')

plt.hist(data['BPMeds'])
plt.title('BPMeds  distribution')
plt.xlabel('BPMeds ')

plt.hist(data['prevalentStroke'])
plt.title('prevalentStroke distribution')
plt.xlabel('prevalentStroke')

plt.hist(data['prevalentHyp'])
plt.title('prevalentHyp distribution')
plt.xlabel('prevalentHyp')

plt.hist(data['diabetes'])
plt.title('diabetes distribution')
plt.xlabel('diabetes')

plt.hist(data['totChol'])
plt.title('totChol distribution')
plt.xlabel('totChol')

plt.hist(data['sysBP'])
plt.title('sysBP distribution')
plt.xlabel('sysBP')

plt.hist(data['diaBP'])
plt.title('diaBP  distribution')
plt.xlabel('diaBP ')

plt.hist(data['BMI'])
plt.title('BMI distribution')
plt.xlabel('BMI')

plt.hist(data['heartRate'])
plt.title('heartRate distribution')
plt.xlabel('heartRate')

plt.hist(data['glucose'])
plt.title('glucose  distribution')
plt.xlabel('glucose')

sns.countplot(x='TenYearCHD',data=data)

data.TenYearCHD.value_counts()
#There are 3,177 patents without heart disease and 572 patients at risk of heart disease.

sns.pairplot(data=data)

correlacion = data.corr()



# Logistic Regression

# independent variable
X = data[['age']]

# dependent varieable
y = data['TenYearCHD']

X.head()

y.head()

from sklearn.linear_model import LogisticRegression

logit_reg = LogisticRegression()
logit_reg.fit(X,y)

# beta_1
logit_reg.coef_

# beta 0
logit_reg.intercept_

### podemos probar otro metodo

X_cons = sm.add_constant(X)

X_cons.head()

import statsmodels.discrete.discrete_model as smd

logit = smd.Logit(y, X_cons).fit()

logit.summary()

### multiple logistic regression

X = data.loc[:,data.columns != 'TenYearCHD']

y = data['TenYearCHD']

mul_lr = LogisticRegression()
mul_lr.fit(X,y)

# beta_1 values
mul_lr.coef_

# beta_0 values
mul_lr.intercept_

X_cons = sm.add_constant(X)

X_cons.head()

logit = smd.Logit(y, X_cons).fit()

logit.summary()

#prediccion y matriz de confucion#

mul_lr.predict_proba(X)

y_pred = mul_lr.predict(X)

y_pred

#estos datos no estan en probabilidades, estan mas bein en clases, 0 o 1 de acuerdo a TenYearCHD 
## ahor hacemos una condicion para 0.3, para los valores mayores al 30%
y_pred_03 = (mul_lr.predict_proba(X)[:,1] >= 0.3)

y_pred_03

#hacemos la matriz de confucion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)
#ahora si la hacemos sobre el 0.3 de la condicion de frontera tendremos
cm3 = confusion_matrix(y,y_pred_03)

#evaluacion del performance del modelo y probabilidad de prediccion
from sklearn.metrics import precision_score, recall_score
precision_score(y, y_pred)
recall_score(y,y_pred)
from sklearn.metrics import roc_auc_score
roc_auc_score(y, y_pred)

##########analisis de discriminante lineal  #############3
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#ahora creamos el modelo
mul_lda = LinearDiscriminantAnalysis()
#lo aplicamos a nuestras variables X y y
mul_lda.fit(X,y)
#ahora vamos a ver la prediccion del modelo lda
y_pred_lda = mul_lda.predict(X)
#desues de esto podemos ver o revisar los resultados que nos da la prediccion
#y ahora la matriz de confusion
lda_cm = confusion_matrix(y, y_pred_lda)


########### division de pruevas y erores para ver el mejor ##
from sklearn.model_selection import train_test_split
#Definimos las variales de entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state = 10)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#ahora hacemos el modelo de regresion logistica
clf_LR =LogisticRegression()
#aplicamos el modelo a X_train y y_train
 clf_LR.fit(X_train, y_train)

#emos las predicciones de las pruevbas de y
y_test_pred = clf_LR.predict(X_test)

#vamoas a hacer la matriz de onfusion y el calificador del modelo

from sklearn.metrics import confusion_matrix, accuracy_score

conf_mat = confusion_matrix(y_test,y_test_pred)

accuracy_score(y_test, y_test_pred)

cm2 = confusion_matrix(y_test, y_test_pred)

conf_matrix=pd.DataFrame(data=cm2,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")

TN=cm2[0,0]
TP=cm2[1,1]
FN=cm2[1,0]
FP=cm2[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)

### Model Evaluation - Statistics

print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',)



















