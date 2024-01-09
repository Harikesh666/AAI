from sklearn import datasets
from sklearn.cluster import KMeans
iris_df = datasets.load_iris()
model = KMeans(n_clusters=3)
model.fit(iris_df.data)
predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])
all_predictions = model.predict(iris_df.data)
predicted_label
all_predictions
