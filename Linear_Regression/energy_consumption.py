import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score


df = pd.read_csv("test_energy_data.csv")
pd.set_option('display.max_columns', None)
print(df)
print("______________________")

#File Reading
print(df.columns)
print("_______________________")
print(df.info())
print("_______________________")
print(df.duplicated())
print("_______________________")
print(df.isnull().sum())
print("_______________________")
print(df['Building Type'].unique())
print("_______________________")

#EDA
sns.lmplot(x = 'Square Footage',y = 'Energy Consumption',data=df)
plt.title("Square Footage vs Energy Consumption")
plt.show()

sns.lmplot(x = 'Number of Occupants',y = 'Energy Consumption',data=df)
plt.title("Number of Occupants vs Energy Consumption")
plt.show()

sns.lmplot(x="Appliances Used", y="Energy Consumption", data=df)
plt.title("Appliances Used vs Energy Consumption")
plt.show()

sns.barplot(x="Day of Week", y="Energy Consumption", data=df)
plt.title("Day of Week vs Energy Consumption")
plt.show()

sns.boxplot(x="Building Type", y="Energy Consumption", data=df)
plt.title("Energy Consumption by Building Type")
plt.show()

#preprocessing
#encoding category to numerical
catog_cols = df[['Building Type','Day of Week']]
for i in catog_cols:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])

#normalising value using standardscaler
std_scale = StandardScaler()
df['Square Footage'] = std_scale.fit_transform(df[['Square Footage']])
print(df)
print("______________________________________")


x = df[['Building Type','Square Footage','Number of Occupants','Appliances Used','Average Temperature','Day of Week']]
y = df['Energy Consumption']

#cross_validation for selecting best model
model = RandomForestRegressor()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
score_cv = cross_val_score(model,x,y,cv=kf,scoring='r2')
print("Random Forest Regressor Score:")
print(score_cv)
print(score_cv.mean())

model_li = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
score_cv = cross_val_score(model_li,x,y,cv=kf,scoring='r2')
print("Linear Regression Score:")
print(score_cv)
print(score_cv.mean())
print("**************************************")

#train and test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)

#Best choice for this dataset is Linear Regression, so proceeding with LinearRegression() model.
my_model = LinearRegression()
my_model.fit(x_train,y_train)

prediction = my_model.predict(x_test)
print(prediction)
print("_______________________")
#r2_score
print("r2_score:",r2_score(y_test,prediction))