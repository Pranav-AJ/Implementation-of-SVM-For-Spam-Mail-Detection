# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: A.J.PRANAV
RegisterNumber: 212222230107
*/
```
```
import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
```
```
import pandas as pd
data= pd.read_csv("/content/spam.csv",encoding='Windows-1252')
```
```
data.head()
```
```
data.info()
```
```
data.isnull().sum()
```
```
x=data["v1"].values
```
```
y=data["v2"].values
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
```
```
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:

### Result output

![image](https://github.com/Pranav-AJ/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118904526/1ad4724c-09df-4d19-ade5-030640f76d19)

### data.head()

![image](https://github.com/Pranav-AJ/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118904526/f0898d80-9138-4154-b3ae-8c163b0320f0)

### data.info()

![image](https://github.com/Pranav-AJ/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118904526/8aebfc32-c9ce-499e-8863-06b0027663e2)

### data.isnull().sum()

![image](https://github.com/Pranav-AJ/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118904526/e408822b-fb8a-4f73-b8d2-cbb810398fff)

### Y_prediction value

![image](https://github.com/Pranav-AJ/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118904526/0a6af7aa-a2e3-4282-9e9e-f4f929fdc4bd)

### Accuracy value

![image](https://github.com/Pranav-AJ/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118904526/ac90dcf0-d46a-46e6-bada-7bd6ef160666)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
