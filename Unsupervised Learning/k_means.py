
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer


########## K-Means ##########
df = pd.read_csv("USArrests.csv", index_col=0)
df.head()

df.isnull().sum()
df.describe().T


sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)

df[0:5]

kmeans= KMeans(n_clusters=4)
k_fit = kmeans.fit(df)


k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_
k_fit.inertia_ ##

df[0:5]

# Visulazation
k_means= KMeans(n_clusters=2).fit(df)
kumeler = k_means.labels_

type(df)
df = pd.DataFrame(df)
plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")
plt.show()

# Cluster Center
centers = k_means.cluster_centers_

plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")

plt.scatter(centers[:, 0],
            centers[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()

# Optimum Cluster Number
kmeans = KMeans()
ssd = []
K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()


kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()
elbow.elbow_value_

# Final Cluster
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
kumeler=kmeans.labels_

df = pd.read_csv("USArrests.csv", index_col=0)
pd.DataFrame({'Eyaletler':df.index, "Kumeler": kumeler})

df["cluster_no"] = kumeler
df["cluster_no"] = df["cluster_no"] + 1

df.head()

df.groupby("cluster_no").agg({"cluster_no": "count"})
df.groupby("cluster_no").agg(np.mean)

df[df["cluster_no"] == 5]
df[df["cluster_no"] == 6]