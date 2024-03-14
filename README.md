# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GOKUL J
RegisterNumber:  212222230038

import pandas as pd

data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1.head()

data1=data1.drop(['sl_no','salary'],axis=1)

data1.isnull().sum()

data1.duplicated().sum()

data1

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

data1

x=data1.iloc[:,:-1]
x

y=data1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

from sklearn.linear_model import LogisticRegression 

model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy = accuracy_score(y_test,y_pred)
confusion = confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfsuion matrix:\n",confusion)
print("\nClassification Report:\n",cr)

from sklearn import metrics
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion,display_labels=[True,False])
cm_display.plot()

```

## Output:
![312095658-762aa859-14c6-4c56-bee7-adfc8a2665c8](https://github.com/Gokul0117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121165938/e50b963c-5c27-4447-90cc-ca6709d72f95)

![312095843-b0a01256-9bec-421c-93e0-fa2e25039be5](https://github.com/Gokul0117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121165938/9a5ff9ca-dd2c-4c30-80ad-a38ca17c33d9)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
