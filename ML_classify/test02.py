from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
X_all=np.array(iris.data)
y_all=np.array(iris.target)

X_train,X_test,y_train,y_test=train_test_split(X_all,y_all,test_size=0.2,random_state=34)

X_1=X_train[:10]
y_1=y_train[:10]
X_2=X_train[110:]
y_2=y_train[110:]

clf=GaussianNB()
clf.partial_fit(X_1,y_1,np.unique(y_1))
y_pred_1=clf.predict(X_test)
acc1=metrics.accuracy_score(y_test,y_pred_1)

clf.partial_fit(X_2,y_2,np.unique(y_2))
y_pred_2=clf.predict(X_test)
acc2=metrics.accuracy_score(y_test,y_pred_2)

print(acc1)
print(acc2)

