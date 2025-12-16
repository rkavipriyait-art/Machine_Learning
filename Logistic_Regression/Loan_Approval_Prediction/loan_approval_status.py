import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import pickle


df = pd.read_csv("loan_data.csv")
pd.set_option("display.max_columns",None)
print(df)
print(df.columns)
print(df.info())
print(df.duplicated().sum())
print(df.isnull().sum())

#finding categorical column with unique data value
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}:{df[col].unique()}")

#EDA
x = df['person_home_ownership']
sns.histplot(x)
plt.xlabel("Home Ownership")
plt.show()

sns.histplot(y = 'loan_intent', data = df)
plt.ylabel("Loan Intension")
plt.show()

sns.scatterplot(x = "credit_score",y = "loan_percent_income", data = df)
plt.xlabel("Credit Score")
plt.ylabel("Loan Status")
plt.show()

sns.barplot(x ='loan_amnt', y ='loan_int_rate', data = df)
plt.xlabel("Loan Amount")
plt.ylabel("Loan Interest Rate")
plt.tight_layout()
plt.show()

plt.figure(figsize=(20,16))
sns.heatmap(final_df.corr(),fmt='.2g',annot=True)
plt.show()

#preprocssing - converting categorical data into numerical

column_to_le = ['person_home_ownership','loan_intent','previous_loan_defaults_on_file','person_gender']

for col in column_to_le:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
print(df)

#preprocessing - feature scaling
col_to_stdscale = ['person_income','loan_amnt','credit_score']

for column in col_to_stdscale:
    std_scale = MinMaxScaler()
    df[column] = std_scale.fit_transform(df[[column]])
print(df)

#preprocessing - one hot encoding for education column
one_hot_encoded_obj = OneHotEncoder(sparse_output=False)
encoded_data = one_hot_encoded_obj.fit_transform(df[['person_education']])
print(encoded_data)

encoded_col_names = one_hot_encoded_obj.get_feature_names_out()
print(encoded_col_names)

encoded_df = pd.DataFrame(encoded_data, columns = encoded_col_names)
print(encoded_df)
print("*******************************************")

df.drop(['person_education'],axis = 1,inplace=True)
final_df = pd.concat([df,encoded_df],axis = 1)
print(final_df)


x = final_df[['person_income','person_home_ownership','credit_score','loan_amnt','loan_intent','person_education_Associate','person_education_Bachelor','person_education_Doctorate','person_education_High School','person_education_Master']]
y = final_df['loan_status']

model = LogisticRegression()

kf = KFold(n_splits=10,shuffle=True,random_state=10)
scores = cross_val_score(model,x,y,cv=kf,scoring='accuracy')
print("cross validation scores:")
print(scores)
print(scores.mean())

model.fit(x,y)
with open("loan_approval.pkl","wb") as file_obj:
    pickle.dump(model,file_obj)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state = 23)
lr_obj = LogisticRegression()
lr_obj.fit(x_train,y_train)

y_pred = lr_obj.predict(x_test)
print(pd.DataFrame(x_test))
print("*********************************")
op_df = pd.DataFrame(y_pred)
pd.set_option('display.max_rows', None)
print(op_df)
#--------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
confusion_matrix = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("confusion matrix:")
print(confusion_matrix)
print("classification_report:")
print(classification_report)





