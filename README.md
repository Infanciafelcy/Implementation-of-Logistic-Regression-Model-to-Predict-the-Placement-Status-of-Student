# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.
```
Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: INFANCIA FELCY P
Register Number: 212223040067
```
   
   

## Program:
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
data1.duplicated().sum()
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:, : -1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
/*

```

## Output:
Placement Data:

![image](https://github.com/user-attachments/assets/9cdd5a76-650b-4b85-98bc-af94831ce849)

Salary Data:

![image](https://github.com/user-attachments/assets/a60952da-254c-49ed-b85a-38c27f943637)

Checking the null factor:

![image](https://github.com/user-attachments/assets/fcd07b64-3a22-4bc5-9c3f-6e058b894118)

Data Dulpicate:

![image](https://github.com/user-attachments/assets/2ed3ec6e-1f13-4df0-af25-69615df10f1a)

Print data:

![image](https://github.com/user-attachments/assets/2ed55e86-29e1-4848-8097-6920e3b90f37)

Data Status:

![image](https://github.com/user-attachments/assets/6551023c-6b9b-4974-a147-b799553b76bc)


Y_Predicition array:

![image](https://github.com/user-attachments/assets/591e82e1-5f68-4341-8479-d05b9f203a9b)

Accuracy Value:

![image](https://github.com/user-attachments/assets/fffac4cd-f8de-4ec4-bd1e-ac566baffdd9)

Confusion array:

![image](https://github.com/user-attachments/assets/069a5921-64a9-4861-bec1-ea6c68dcb127)

Classification Report:

![image](https://github.com/user-attachments/assets/f084bad9-ca8c-476e-82fb-cb36636e1853)

Prediction of LR:

![image](https://github.com/user-attachments/assets/e20295a6-bdea-43fb-842b-6269c9807ad3)






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
