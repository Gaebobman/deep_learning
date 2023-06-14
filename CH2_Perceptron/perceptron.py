import numpy as np

x = np.array([0, 1])        # Input
w = np.array([0.5, 0.5])    # Weight
b = -0.7                    # bias

print(w*x)
print(np.sum(w*x))
print(np.sum(w*x) + b)


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2],)
    w = np.array([-0.5, -0.5])  # AND 와 Weight 와 Bias만 다르다.
    b =  0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2    
    tmp = np.sum(w * x) + b
    if tmp <=0:
        return 0
    else:
        return 1
    
# XOR은 선형적으로 구현이 불가능 하다.
# Multi-layer perceptron 으로는 구현 가능하다.

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))
