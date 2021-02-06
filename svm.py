# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:30:29 2021

@author: lucho
"""

import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.svm import SVC
import sklearn.metrics as sm


iris = datasets.load_iris()

print(iris)
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)

x=pd.DataFrame(iris.data)
# définir les noms de colonnes
x.columns=['Sepal_Length','Sepal_width','Petal_Length','Petal_width']
y=pd.DataFrame(iris.target)
y.columns=['Targets']
#########################################################################
new_target =  np.where(iris.target<1, 0, 1)
colormap =np.array(['BLUE','GREEN','CYAN'])
plt.scatter(x.Sepal_Length, x.Sepal_width,c=colormap[new_target],s=40)

new_data = x[['Sepal_Length','Sepal_width']]
plt.scatter(new_data.Sepal_Length, new_data.Sepal_width,
            c=colormap[new_target],s=40)


svm = SVC(kernel='linear')
svm.fit(new_data, new_target)
svm.support_vectors_

# 1. Tracer l'hyperplan de marge maximale séparateur
#calcul des coordonnés de deux points qui passent par la droite 

#frontiere f(x) = W'x+b = 1
xh = np.array([4,8])
yh = -svm.coef_[0][0]/svm.coef_[0][1]*xh-(svm.intercept_+1.0)/svm.coef_[0][1]
#frontiere f(x) = W'x+b = -1
xb = np.array([4.5,8])
yb = -svm.coef_[0][0]/svm.coef_[0][1]*xb-(svm.intercept_-1.0)/svm.coef_[0][1]

#frontiere f(x) = W'x+b = 0
xf = np.array([3,7])
yf = -svm.coef_[0][0]/svm.coef_[0][1]*xf-svm.intercept_/svm.coef_[0][1]

plt.scatter(new_data['Sepal_Length'], new_data['Sepal_width'], c = colormap[new_target], s = 40)
plt.plot(xf,yf,c='green')
plt.plot(xb,yb,c='gray')
plt.plot(xh,yh,c='gray')
plt.show()

# 2. Evaluer l'algorithme de classification
svm.score(new_data,new_target)
svm.predict(new_data)
sm.confusion_matrix(new_target,svm.predict(new_data))
sm.plot_confusion_matrix(svm,new_data,new_target)  

# 3. Choisir le C optimal en utilisant la Validation Croisée 
from sklearn.model_selection import GridSearchCV

params = {'C': np.linspace(1e-2,1e2,100)}

gs = GridSearchCV(SVC(), params, cv=10)
gs.fit(new_data, new_target)

print(gs.best_params_)