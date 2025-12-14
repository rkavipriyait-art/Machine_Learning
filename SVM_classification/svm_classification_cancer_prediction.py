import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data = pd.read_csv('Cancer_Data.csv')
#pd.set_option('display.max_columns',None)
print(data)
print("___________________________")

#File Reading
print(data.info())
print("___________________________")
print(data.describe())
print("___________________________")
print(data.columns)
print("___________________________")
print(data.duplicated())
print("___________________________")
print(data.isnull().sum())
data.drop('Unnamed: 32', axis=1, inplace=True)
print("___________________________")
print(data.isnull().sum())
print("___________________________")

#EDA
sns.boxplot(x="diagnosis", y="radius_mean", data=data)
plt.show()
sns.violinplot(x="diagnosis", y="texture_mean", data=data)
plt.show()
sns.scatterplot(x="radius_mean", y="texture_mean", hue="diagnosis", data=data)
plt.show()
sns.histplot(data=data, x="area_worst", hue="diagnosis", kde=True)
plt.show()
sns.pairplot(data, vars=["radius_mean","area_mean","perimeter_mean"], hue="diagnosis")
plt.show()

x = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

#cross validation to select best model
models = {
    "SVM": SVC(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

for name, model in models.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, x, y, cv=skf, scoring='accuracy')
    print(f"{name} Average Accuracy: {np.mean(scores)}")
print("___________________________")


#train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Choosen RandomForestClassifier() based on cross_val_score
my_model = RandomForestClassifier(n_estimators=10)
my_model.fit(x_train,y_train)

y_pred = my_model.predict(x_test)
print(y_pred)
print("___________________________")
print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("Classification Report:")
print(classification_report(y_test,y_pred))