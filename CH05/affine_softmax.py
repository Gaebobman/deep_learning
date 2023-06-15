import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x      # dot operation 역전파 계산시 필요하기에 저장
        out = np.dot(x, self.W) + self.b

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx
    


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)   # a - c 를 함으로써 Overflow를 방지한다.
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    delta = 1e-7        # log에 0 들어가면 -inf로 계산 불가
    batch_size  = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size       


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t) 
        return self.loss

    def backward(self, dout=1):
        # 주의: 역전파시에는 전파하는 값을 배치의 수로 나눠서 
        # 데이터 1개당 오차를 앞 계층으로 전파
        
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx