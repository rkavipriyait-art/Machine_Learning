import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report


df = pd.read_csv("diabetes_prediction_dataset.csv")
pd.set_option("display.max_columns",None)
print(df)
print("___________________________")
print(df.columns)
print("___________________________")
print(df.describe())
print("___________________________")
print(df.info())

#EDA
plt.figure(figsize=(8,6))
sns.barplot(x="gender", y="hypertension",data=df)
plt.show()

plt.figure(figsize=(8,6))
sns.lineplot(x="HbA1c_level", y="diabetes",data=df)
plt.show()

plt.figure(figsize=(8,6))
sns.lineplot(x="blood_glucose_level", y="diabetes",data=df)
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(x="gender", y="diabetes",data=df)
plt.show()

X = df[["hypertension","heart_disease","bmi","HbA1c_level","blood_glucose_level"]]
Y = df["diabetes"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state=2)

#feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print("_____________________________________")
logistic_reg_obj = LogisticRegression()
logistic_reg_obj.fit(X_train,Y_train)

#Prediction
Y_pred = logistic_reg_obj.predict(X_test)
output_df = pd.DataFrame(X_test,Y_pred)
print(output_df)
print(output_df[110:125])

print("_____________________________________")
final_accuracy_check = accuracy_score(Y_test, Y_pred)
print(final_accuracy_check)
print("classification_report:")
print(classification_report(Y_pred,Y_test))





