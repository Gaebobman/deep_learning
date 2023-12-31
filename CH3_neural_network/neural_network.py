import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# softmax function
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)   # a - c 를 함으로써 Overflow를 방지한다.
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
    


# 항등함수
def identify_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])    # 가중치 1 초기화
    network['b1'] = np.array([0.1, 0.2, 0.3])                       # 편향 1 초기화

    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


# forward propagation
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identify_function(a3)

    return(y)

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)



a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))        # Softmax 출력의 총 합은 1이다. 따라서 확률의 개념으로 이해할 수 있음