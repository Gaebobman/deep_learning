# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch03/neuralnet_mnist.py
# coding: utf-8
from mnist import load_mnist
import pickle
import numpy as np
import sys
import os

path = os.getcwd()
sys.path.append(os.path.abspath(path))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)   #  Normalize 사용하여 Preprocessing 
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()       # MNIST Dataset을 불러옴
network = init_network()    # 네트워크를 생성

batch_size = 100
accuracy_cnt = 0

''' 
# With out Batch
for i in range(len(x)):
    y = predict(network, x[i])  # x에 저장된 이미지 하나씩 꺼내 predict()
    p = np.argmax(y)            # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1
'''
# with batch

for i in  range(0 , len(x), batch_size):
    x_batch = x[i: i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i: i+batch_size])

print("Accuracy:"+str(float(accuracy_cnt) / len(x)))