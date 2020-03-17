from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


'''0 - Extremely Weak

1 - Weak

2 - Normal

3 - Overweight

4 - Obesity

5 - Extreme Obesity'''
ds = pd.read_csv('datasets/500_Person_Gender_Height_Weight_Index.csv')

ds['Gender'] = ds['Gender'].map({'Male': 0, 'Female': 1})
#ds.head()

set1 = ds.drop('Index', axis=1)
set2 = ds['Index']

X_train, X_test, y_train, y_test = train_test_split(set1, set2, random_state=42)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

#testcases
predict = clf.predict([[1,167.64,62]])


#print(accuracy_score(y_test, y_predict))