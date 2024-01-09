import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

X = np.array([1, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1])
y = np.array([2, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 3, 8.8, 7.5])

plt.scatter(X, y)
plt.show()

training_X = np.vstack((X, y)).T
print(np.vstack((X, y)).T)
training_y = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]

clf = svm.SVC(kernel='linear', C=1.0)

clf.fit(training_X, training_y)

w = clf.coef_[0]

a = -w[0] / w[1]

XX = np.linspace(0, 13)

yy = a * XX - clf.intercept_[0] / w[1]

plt.plot(XX, yy, 'k-')

plt.scatter(training_X[:, 0], training_X[:, 1], c=training_y)
plt.legend()
plt.show()
