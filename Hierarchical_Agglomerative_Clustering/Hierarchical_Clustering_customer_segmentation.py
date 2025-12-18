import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

df = pd.read_csv("Mall_Customers.csv")
print(df)
print("********************************")

print(df.info())
print("********************************")
print(df.columns)
print("********************************")
print(df.isnull().sum())


# Select features
x = df.select_dtypes(include=['number'])

# Scale data (important for cosine)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#to find n_cluster value(k)
model = AgglomerativeClustering()
visualizer = KElbowVisualizer(model, k=(2,10))
visualizer.fit(x_scaled)
visualizer.show()


silhouette_scores = []
k = range(2, 11)

for i in ["euclidean", "manhattan", "cosine"]:
    clustering = AgglomerativeClustering(
        n_clusters=5,
        metric=i,
        linkage="average")
    labels = clustering.fit_predict(x_scaled)
    score = silhouette_score(x_scaled, labels)
    silhouette_scores.append(score)

    print(f"metric = {i}, Silhouette Score = {score:.4f}")

df['Cluster'] = labels
print(df)

plt.scatter(
    x_scaled[:, 0],
    x_scaled[:, 1],
    c=labels,
    cmap='rainbow'
)
plt.xlabel(x.columns[0])
plt.ylabel(x.columns[1])
plt.title("Agglomerative Clustering")
plt.show()

linked = linkage(x_scaled, method='average')

plt.figure(figsize=(10, 6))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()