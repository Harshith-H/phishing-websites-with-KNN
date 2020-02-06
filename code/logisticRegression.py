import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

data=pd.read_csv('phishing.txt')

#print(data)
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
#print(y)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)


lr=LogisticRegression(C=100,random_state=0)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

print('Test Accuracy=')
print(accuracy_score(y_test,y_pred))
print()
print(confusion_matrix(y_test,y_pred))
print()

y_pred=lr.predict(X_train)

print('Training Accuracy=')
print(accuracy_score(y_train,y_pred))
print()
print(confusion_matrix(y_train,y_pred))
