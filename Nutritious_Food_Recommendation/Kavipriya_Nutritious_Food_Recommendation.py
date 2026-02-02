import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer

#FILE READING
df = pd.read_csv("Kavipriya_Nutritious_Food_Recommendation.csv")
#pd.set_option('display.max_columns',None)
print(df)
print("_______________________________________")

print(df.shape)
print("_________________________________")
print(df.dtypes)
print("_________________________________")
print(df.describe())
print("_________________________________")
print(df.info())
print("_________________________________")
print(df.duplicated())
print("_________________________________")

#For Columns
print(df.columns)
print("_________________________________")
print(df['Daily Calorie Target'])
print("_________________________________")
print(df[['Ages','Activity Level']])
print("_________________________________")

#For Rows
print(df.head())
print("_________________________________")
print(df.tail())
print("_________________________________")
print(df[100:120])
print("_________________________________")
print(df.loc[10:25])
print("_________________________________")

#For Rows and Columns
print(df.loc[800:830,"Disease"])
print("_________________________________")
print(df.loc[500:520,['Daily Calorie Target','Protein']])
print("_________________________________")


#PREPROCESSING

print(df.isnull().sum())
print("_________________________________")
print(df["Dietary Preference"].value_counts())
print("_________________________________")
print(df['Dietary Preference'].unique())
print("_________________________________")

#EDA

#age distribution
plt.figure(figsize=(6,4))
sns.histplot(df["Ages"], bins=10, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

#Average Calories by Dietary Preference
plt.figure(figsize=(6,4))
sns.barplot(
    x="Dietary Preference",
    y="Calories",
    data=df,
    estimator="mean"
)
plt.title("Average Calories by Dietary Preference")
plt.xticks(rotation=20)
plt.show()

#Correlation heatmap
cols = [
    "Ages", "Height", "Weight", "Calories",
    "Protein", "Carbohydrates", "Fat", "Sugar", "Sodium", "Fiber"
]
corr = df[cols].corr()
plt.figure(figsize=(9,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Calculate BMI (Height in cm, Weight in kg)
df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)

# Define BMI category
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Apply BMI category
df["BMI Category"] = df["BMI"].apply(bmi_category)
print(df)
print("_________________________________")
print(df.info())
print("_________________________________")

#converting categorical to numerical column using Label Encoder
label_encoders = {}
categorical_column = ['Gender','Activity Level','Disease','BMI Category']

for col in categorical_column:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print(df[['Gender','Activity Level','Disease','BMI Category']])
print("_________________________________")

# CLUSTERING
# Select Nutrition Features
features = df[["Calories", "Protein", "Carbohydrates", "Fat"]]

# Normalize Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Finding best K-value for kmeans clustering using yellowbrick.cluster(library) -> KElbowVisualizer
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(
    model,
    k=(2,10),
    metric="distortion",
    timings=False
)
visualizer.fit(scaled_features)
visualizer.show()

#Kmeans Clustering Usage - Foods are clustered based on nutritional similarity, and recommendations are generated from the most relevant cluster
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_features)
print(df["Cluster"])

"""KMeans clustering is performed in multi-dimensional space (4 features in your case: Calories, Protein, Carbohydrates, Fat).
PCA (Principal Component Analysis) reduces high-dimensional data into 2 principal components while preserving maximum variance,
making cluster visualization possible. So that i'm using PCA-based scatter plot for better cluster output"""

# Get labels and cluster centers from KMeans
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Apply PCA (2D)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_features)
pca_centers = pca.transform(centers)

# Plot
plt.figure(figsize=(7,5))
plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.7
)

# Plot cluster centers
plt.scatter(
    pca_centers[:, 0],
    pca_centers[:, 1],
    marker="X",
    s=200,
    linewidths=2
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("KMeans Clustering (PCA Projection)")
plt.show()
print("________________________________")

#cluster_summary used for better understanding what each cluster represents
cluster_summary = df.groupby("Cluster")[["Calories", "Protein", "Carbohydrates", "Fat"]].mean()
print(cluster_summary)
print("________________________________")
# cluster_map is used to convert technical cluster IDs into human-readable categories based on cluster_summary output
cluster_map = {
    0: "High Carb",
    1: "Energy Dense",
    2: "Low Calorie",
    3: "High Protein"
}
df["Cluster_Type"] = df["Cluster"].map(cluster_map)
print(df)

df["Cluster_Type"].value_counts().plot(kind="bar")
plt.xlabel("Cluster Type")
plt.ylabel("Count")
plt.title("Distribution of Nutrition Clusters")
plt.tight_layout()
plt.show()


# Cluster Preference per BMI
bmi_cluster_pref = {
    "Underweight": ["Energy Dense", "High Carb"],
    "Normal": ["High Protein", "High Carb"],
    "Overweight": ["Low Calorie", "High Protein"],
    "Obese": ["Low Calorie"]
}

pd.crosstab(df["BMI Category"], df["Cluster_Type"]).plot(
    kind="bar", stacked=True, figsize=(10,6)
)
plt.ylabel("Count")
plt.title("BMI Category vs Nutrition Cluster")
plt.show()

#Assuming BMI category based daily calory target
daily_calorie_target = {
    "Underweight": "2800–3000 kcal/day",
    "Normal": "2000–2200 kcal/day",
    "Overweight": "1600–1800 kcal/day",
    "Obese": "1200–1500 kcal/day"
}

#EDA
# BMI categories and calorie targets (midpoints)
bmi_categories = ["Underweight", "Normal", "Overweight", "Obese"]
calorie_targets = [2900, 2100, 1700, 1350]
plt.figure()
plt.bar(bmi_categories, calorie_targets)
plt.xlabel("BMI Category")
plt.ylabel("Daily Calorie Target (kcal/day)")
plt.title("Daily Calorie Target by BMI Category")
plt.show()

"""Recommends meal suggestions based on:
1.BMI category (mapped to nutrition clusters)
2.Dietary preference (optional filter)
This is a rule-based & cluster-based hybrid recommender, which is very appropriate for nutrition systems."""

def recommend_meals(df, bmi_cat, dietary_pref=None):
    # Get clusters recommended for this BMI category
    preferred_clusters = bmi_cluster_pref.get(bmi_cat, [])

    # Filter foods by preferred clusters, Keeps only rows whose Cluster_Type matches the recommended clusters
    recommendations = df[df["Cluster_Type"].isin(preferred_clusters)]

    # If dietary preference is provided, filter meals
    if dietary_pref:
        # Assuming dietary preference is also labeled in the meal columns
        mask = (
            (recommendations["Dietary Preference"].str.contains(dietary_pref, case=False, na=False))
        )
        recommendations = recommendations[mask]

    # Get unique meal suggestions
    breakfast = recommendations["Breakfast Suggestion"].dropna().unique().tolist()
    lunch = recommendations["Lunch Suggestion"].dropna().unique().tolist()
    dinner = recommendations["Dinner Suggestion"].dropna().unique().tolist()
    snacks = recommendations["Snack Suggestion"].dropna().unique().tolist()

    return {
        "Breakfast": breakfast,
        "Lunch": lunch,
        "Dinner": dinner,
        "Snacks": snacks
    }

#Sample output check
example_bmi_cat = "Underweight"
example_diet = "Omnivore"
meals = recommend_meals(df, example_bmi_cat, example_diet)

print("Meal Recommendations for BMI:", example_bmi_cat, "\nDiet:", example_diet)
print("Daily Calorie Target:", daily_calorie_target.get(example_bmi_cat, "Not Defined"))
print("Breakfast:", meals["Breakfast"])
print("Lunch:", meals["Lunch"])
print("Dinner:", meals["Dinner"])
print("Snacks:", meals["Snacks"])


import pickle

# Assuming df is your preprocessed DataFrame with clusters
with open("nutrition_df.pkl", "wb") as f:
    pickle.dump(df, f)
