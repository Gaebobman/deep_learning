import numpy as np


class Relu:
    def __init__(self):
        # True / False로 구성된 넘파이 배열
        self.mask = None   

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        # 0 이하면 0으로 mask
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
    
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

mask = (x <= 0)
print(mask)

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out      # 역전파시 사용하기 위해 순전파 출력을 저장한다.
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        
        return dx
    


