from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
type(iris)
iris.keys()
type(iris.data), type(iris.target)
iris.data.shape
iris.target_names
X = iris.data
Y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])
X = [
    [5.9, 1.0, 5.1, 1.8],
    [3.4, 2.0, 1.1, 4.8],
]
X
prediction = knn.predict(X)
prediction
