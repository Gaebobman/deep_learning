import numpy as np
import matplotlib.pyplot as plt

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)

for row in X:
    print(row)

X = X.flatten()
print(X)
print(X > 15)
print(X[X > 15])


x = np.arange(0, 6, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()
