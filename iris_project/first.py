from nbformat import read
import pandas as pd
from pandas.plotting   import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from  sklearn.model_selection  import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(filename,names = names)
#print(dataset.shape)
#print(dataset.describe())
#print(dataset.groupby('class').size())

#univariate plots
"""
dataset.plot(kind = 'box',subplots = True,layout =(2,2),sharex=False, sharey=False)
plt.show()
           """
#histogram

"""
dataset.hist()
plt.show()
"""
#multivariate plots

"""scatter_matrix(dataset)
plt.show()
"""
#split out validation dataset
array =  dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed =7
X_train ,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size=validation_size,random_state =seed)


models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#evaluate each model in turn 
result = []
names = []

for name,model in models:
    kfold = KFold(n_splits=10,random_state=None)
    cv_results = cross_val_score(model,X_train,Y_train,cv = kfold,scoring ='accuracy')
    result.append(cv_results)
    names.append(name)
    msg = f"{name} {cv_results.mean():.2f} {cv_results.std():.3f}"
    print(msg)