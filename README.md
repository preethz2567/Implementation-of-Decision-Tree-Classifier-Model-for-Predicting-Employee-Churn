# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Import required libraries**  
   Use `pandas` for data handling, `LabelEncoder` for encoding categorical variables, and `DecisionTreeClassifier` from `sklearn`.

2. **Load the dataset**  
   Read the employee data CSV file using `pd.read_csv()`.

3. **Explore the data**  
   Use `.head()`, `.tail()`, and `.isnull().sum()` to understand the structure and check for missing values.

4. **Encode categorical columns**  
   Use `LabelEncoder` to convert non-numeric columns like `salary` into numeric values.

5. **Define features and target**  
   - `x`: Features like satisfaction level, project count, hours, etc.  
   - `y`: Target variable (`left` – whether the employee left or stayed).

6. **Split the dataset**  
   Use `train_test_split()` to divide the data into training and testing sets (e.g., 80% train, 20% test).

7. **Initialize the model**  
   Create a `DecisionTreeClassifier` with `criterion="entropy"` for information gain.

8. **Train the model**  
   Fit the classifier using the training data (`x_train`, `y_train`).

9. **Predict using the model**  
   Predict the output for the test set using `predict()`.

10. **Evaluate the model**  
   Calculate accuracy using `metrics.accuracy_score()` and optionally display predictions.

## Program and Output:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: PREETHI D
RegisterNumber:  212224040250
```

```
import pandas as pd
data=pd.read_csv("Employee.csv")
```
```
data.head()
```
![image](https://github.com/user-attachments/assets/3e7275b3-08e2-4af0-adb1-f816b827dcee)

```
data.tail()
```
![image](https://github.com/user-attachments/assets/1d08979b-1638-480d-b931-cf14ce8bf159)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/3f19fa22-0812-4996-ba03-8ca9d3164803)

```
data["left"].value_counts()
```
![image](https://github.com/user-attachments/assets/571ec76f-7c08-4659-be32-d0fc24dc02a6)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
![image](https://github.com/user-attachments/assets/fa3b370d-9610-4c67-90d9-dc41c606f345)

```
x=data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years","salary"]]
x
```
![image](https://github.com/user-attachments/assets/22461742-f4ae-4790-821c-6d75aa4fa40a)
```
y=data["left"]
y
```
![image](https://github.com/user-attachments/assets/2cc8573f-5ee6-4c7d-9abf-221f530a98f3)

```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=100)
```
```
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test, y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/935d4e4e-b6d9-41cc-9dca-44b23b1a9e32)
```
dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
```
![image](https://github.com/user-attachments/assets/b8ef7ce4-a421-43d6-b78a-2c3e9ba74b64)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
