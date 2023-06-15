import numpy as np
from mnist import load_mnist
from SGD import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)


network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# Hyper Parameters
iters_num = 10000   # 반복횟수
train_size = x_train.shape[0]
batch_size = 100    # size of mini batch
learning_rate = 0.1

train_loss_list = []    # 학습 경과를 기록함
train_acc_list = []   
test_acc_list = []

# 1 Epoch 당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | "
            + str(train_acc) + ", " + str(test_acc))
