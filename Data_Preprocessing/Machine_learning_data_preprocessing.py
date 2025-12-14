import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
#pd.set_option('display.max_columns', None)
print(df)

#Analysing data
print("dataset info():")
print(df.info())
print("*********************************")
print("describe() about dataset")
print(df.describe())
print("*********************************")
print("no.of rows and columns in dataset:")
print(df.shape)
print("*********************************")
print("column names in dataset:")
print(df.columns)
print("*********************************")
print("number of duplicates:")
print(df.duplicated())
print("*********************************")

#preprocessing steps

#finding null values
print("Null Value counts:")
print(df.isnull().sum())
print("*********************************")

#filling string null values using forward fill
df.ffill(inplace=True)

#using SimpleImputer going to fill numerical null values
num_col = df[['MonthlyRate','HourlyRate','DistanceFromHome']]

for i in num_col:
    si_obj = SimpleImputer(strategy='most_frequent')
    df[i] = si_obj.fit_transform(df[[i]])

print("Final null value count:")
print(df.isnull().sum().sum())
print("*********************************")

#finding unique values in categorical column
print("Unique values appears in JobRole column")
print(df['JobRole'].unique())
print("*********************************")
#finding counts of unique value that appeared in particular column
print("total count of each unique values appears in JobRole column")
print(df['JobRole'].value_counts())
print("*********************************")

#Converting Categorical columns into numerical columns using Onehot and Ordinal encoder
#Feature scaling done by Minmaxscaler

ohe_cols = ['JobRole', 'EducationField']
ord_enc_cols = ['BusinessTravel','Department','Gender','OverTime']
scaling_cols = ['DailyRate','MonthlyIncome','MonthlyRate']

# Create the ColumnTransformer
ct = ColumnTransformer(
    transformers=[
        ('one_hot_enc_obj', OneHotEncoder(sparse_output=False, drop=None), ohe_cols),
        ('ordinal_enc_obj',OrdinalEncoder(), ord_enc_cols),
        ('scaler', MinMaxScaler(), scaling_cols)
    ],
    remainder='passthrough'  # Keep all other columns as they are
)

# Fit and transform the data
transformed_data = ct.fit_transform(df)
print(transformed_data)
print("******************************************")
#Get the encoded column names
encoded_column_names = ct.named_transformers_['one_hot_enc_obj'].get_feature_names_out(ohe_cols)
all_column_names = list(encoded_column_names) + ord_enc_cols + scaling_cols + [col for col in df.columns if col not in (ohe_cols + ord_enc_cols + scaling_cols)]
encoded_df = pd.DataFrame(transformed_data, columns=all_column_names)
updated_df = df.drop(['JobRole', 'EducationField'], axis=1)
print("Final Dataframe of the preprocessed dataset:")
print(encoded_df)
print("***********************************************")


# final_df = pd.concat([updated_df,encoded_df],axis=1)
# print(final_df)



