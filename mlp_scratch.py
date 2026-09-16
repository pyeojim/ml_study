import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

X = X.astype(np.float32) / 255.0

def relu(x):
    return np.maximum(0, x)

def reluMask(x, y):
    return x * (y>0)

def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis =1, keepdims=True)

def crossEntropy(p, y):
    return -np.mean(
        np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))
    )

class layer:
    def __init__(self, input_size, output_size):
        self.w = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size).astype(np.float32)
        self.b = np.zeros(output_size, dtype=np.float32)
        self.w_grad = 0
        self.b_grad = 0

# let final loss M. Then, dM/dw = dM/dz * dz/dw = grad * input
# dM/db = dM/dz * dz/db = grad * 1
# dM/dx = dM/dz * dz/dx = grad * w

    def mul(self, input):
        self.input = input
        return input @ self.w + self.b

    def backward(self, grad):
        self.w_grad = self.input.T @ grad
        self.b_grad = np.sum(grad, axis=0)
        return grad @ self.w.T

    def update(self, lr):
        self.w -= lr * self.w_grad
        self.b -= lr * self.b_grad

l1 = layer(X.shape[1], 128)
l2 = layer(128, 128)
l3 = layer(128, 10)

for _ in range(100):
    z1 = l1.mul(X)
    a1 = relu(z1)

    z2 = l2.mul(a1)
    a2 = relu(z2)

    z3 = l3.mul(a2)
    p = softmax(z3)

    LR = 3e-1

    l1.backward(reluMask(l2.backward(reluMask(l3.backward((p - np.eye(10, dtype=np.float32)[y]) / len(y)), z2)), z1))

    l1.update(LR)
    l2.update(LR)
    l3.update(LR)

    pred = np.argmax(p, axis=1)
    acc = np.mean(pred == y)

    print("loss:", crossEntropy(p, y), "acc:", acc)


