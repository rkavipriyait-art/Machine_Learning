import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score,KFold,train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


df = pd.read_csv('cardekho_data.csv')
pd.set_option('display.max_columns',None)
print(df)

print("Column Names:",df.columns)

print(df.info())

print('unique values in Car_Name col:')
print(df['Car_Name'].unique())
print(df['Car_Name'].nunique())

print('Null values count:')
print(df.isnull().sum())


#EDA
plt.figure()
sns.histplot(df['Selling_Price'], kde=True)
plt.title('Distribution of Selling Price')
plt.show()

plt.figure()
sns.boxplot(x=df['Selling_Price'])
plt.title('Outliers in Selling Price')
plt.show()

plt.figure(figsize=(10,8))
corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#Outliers detection
Q1 = df['Selling_Price'].quantile(0.25)
Q3 = df['Selling_Price'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[
    (df['Selling_Price'] < Q1 - 1.5 * IQR) |
    (df['Selling_Price'] > Q3 + 1.5 * IQR)
]
print("Dataset with outlier:", outliers)

df = df[
    (df['Selling_Price'] > Q1 - 1.5 * IQR) |
    (df['Selling_Price'] < Q3 + 1.5 * IQR)
]
print("Dataset shape after outlier removal:", df.shape)

#Encoding categorical to numerical
col_to_encode = df[['Car_Name','Fuel_Type','Seller_Type','Transmission']]

for i in col_to_encode:
    ord_enc = OrdinalEncoder()
    df[i] = ord_enc.fit_transform(df[[i]])
print(df)

x = df.drop(columns=['Selling_Price'])
y = df['Selling_Price']


#cross_valiidation
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = LinearRegression()
score = cross_val_score(model,x,y,cv=kf,scoring='r2')
print("Linear Regression Score:")
print(score)

model = DecisionTreeRegressor()
score = cross_val_score(model,x,y,cv=kf,scoring='r2')
print("Decision Tree Regressor Score:")
print(score)

model = RandomForestRegressor()
score = cross_val_score(model,x,y,cv=kf,scoring='r2')
print("Random Forest Regressor Score:")
print(score)

#Getting high scoring in Random Forest Regressor

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=20)

rf = RandomForestRegressor()

param_grid = {
    'n_estimators' : [10,20,30],
    'criterion': ['squared_error', 'absolute_error'],
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3]
     }


grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(x_train, y_train)
best_est = grid_search.best_estimator_
y_pred = best_est.predict(x_test)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nR² Score:")
print(r2_score(y_test, y_pred))
