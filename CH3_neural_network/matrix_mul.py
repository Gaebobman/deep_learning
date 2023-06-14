import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A.shape, B.shape)
print(np.dot(A, B))

C = np.array([[1, 2, 3], [4, 5, 6]])
D = np.array([[1, 2], [3, 4], [5, 6]])
print(np.dot(C, D), '\n###\n\n\n###')

# Matrix Multiplication in Neural network

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)
print(Y)


X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape, X.shape, B1.shape)

A1 = np.dot(X, W1) + B1     # 1층
print(A1)
Z1 = sigmoid(A1)       # Sigmoid 변환 / Z = h(x)한 값 

W2 = np.array([[0.1 , 0.4],[0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)       # 2층의 Sigmoid 변환

# 항등함수
def identify_function(x):
    return X

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identify_function(A3)   