#Train acc
#Test acc
#Recall
#Precision
#F1
#ROC-AUC
#Fit Diagnosis
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

X = df.drop(['Patient_ID','Gender','Smoking_Status','Alcohol_Consumption','Heart_Failure_Type','Readmitted_30_Days'],axis=1)
y = df['Readmitted_30_Days']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test,y_test_pred)
train_acc = accuracy_score(y_train,y_train_pred)
print(f'Train Acc:{train_acc}')
print(f'Test Acc:{test_acc}')

from sklearn.metrics import precision_score
precision = precision_score(y_test,y_test_pred)
print(f'Precision:{precision}')

from sklearn.metrics import recall_score
recall = recall_score(y_test,y_test_pred)
print(f'Recall:{recall}')

from sklearn.metrics import f1_score
F1 = f1_score(y_test,y_test_pred)
print(f'F1:{F1}')

from sklearn.metrics import roc_auc_score
ROC_AUC = roc_auc_score(y_test,y_test_pred)
print(f'ROC-AUC:{ROC_AUC}')

