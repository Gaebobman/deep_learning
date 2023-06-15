# Stochastic gradient descent
import numpy as np
import sys
import os
from functions import *
from gradient_simplenet import numerical_gradient

path = os.getcwd()
sys.path.append(os.path.abspath(path))


class TwoLayerNet:

    # 초기화: 입력층의 뉴런 수 , 은닉층의 뉴런 수 , 출력츠의 뉴런 수
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}    # 신경망의 매개변수를 보관하는 딕셔너리
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)    # 1번째 층의 가중치
        self.params['b1'] = np.zeros(hidden_size)       # 1번째 층의 편향

        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)   # 2번째 층의 가중치
        self.params['b2'] = np.zeros(output_size)       # 2번째 층의 편향

    # 예측을 수행, x는 이미지 데이터
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # 손실함수의 값을 구함, x는 이미지 데이터, t는 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    # 정확도 계산 x는 이미지 데이터, t는 정답 레이블
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 가중치 매개변수의 기울기를 구함, x는 이미지 데이터, t는 정답 레이블
    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])     # 1번째 층의 가중치의 기울기
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])     # 1번째 층의 편향의 기울기
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])     # 2번째 층의 가중치의 기울기
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])     # 2번째 층의 편향의 기울기

        return grads
    


net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)

x = np.random.rand(100, 784)
t = np.random.rand(100, 10)

grads = net.numerical_gradient(x, t)

print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)
