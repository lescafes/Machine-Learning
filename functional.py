import numpy as np
import torch
from modules import Module
class Sigmoid(Module):
    def forward(self, x):
        # TODO Implement forward propogation
        # of sigmoid function.

        self.y = 1 / (1 + np.exp(-x))
        return self.y
        ...

        # End of todo

    def backward(self, dy):
        # TODO Implement backward propogation
        # of sigmoid function.

        return dy * self.y * (1 - self.y)
        ...

        # End of todo

class Tanh(Module):
    def forward(self, x):
        # TODO Implement forward propogation
        # of tanh function.

        self.y = np.tanh(x)
        return self.y
        ...

        # End of todo

    def backward(self, dy):
        # TODO Implement backward propogation
        # of tanh function.

        return dy * (1 - self.y ** 2)
        ...

        # End of todo

class ReLU(Module):
    def forward(self, x):
        # TODO Implement forward propogation
        # of ReLU function.

        self.x = x
        return np.maximum(x, 0)
        ...

        # End of todo

    def backward(self, dy):
        # TODO Implement backward propogation
        # of ReLU function.

        return np.where(self.x > 0, dy, 0)
        ...

        # End of todo

class Softmax(Module):
    def forward(self, nx):
        # TODO Implement forward propogation
        # of Softmax function.
        """exps = np.exp(x)
        self.solfmax=exps / np.sum(exps, axis=1, keepdims=True)"""
        shifted_x = nx - np.max(nx)
        ex = np.exp(shifted_x)
        sum_ex = np.sum(ex)
        self.solfmax = ex / sum_ex
        return self.solfmax
        ...

        # End of todo
    def get_grad(self):
            self.grad = self.solfmax[:, np.newaxis] * self.solfmax[np.newaxis, :]
            for i in range(len(self.grad)):
                self.grad[i, i] -= self.solfmax[i]
            self.grad = - self.grad
            return self.grad

    def backward(self, dy):
        # Omitted.
        self.get_grad()
        self.dnx = np.sum(self.grad * dy, axis=1)
        return self.dnx
        ...
class Loss(object):
    """
    Usage:
        >>> criterion = Loss(n_classes)
        >>> ...
        >>> for epoch in n_epochs:
        ...     ...
        ...     probs = model(x)
        ...     loss = criterion(probs, target)
        ...     model.backward(loss.backward())
        ...     ...
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
    def __call__(self, probs, targets):
        self.probs = probs
        self.targets = targets
        ...
        return self
    def backward(self):
        ...
class SoftmaxLoss(Loss):
    def __call__(self, probs, targets):

        # TODO Calculate softmax loss.

        #super(SoftmaxLoss, self).__call__(probs, targets)

        ...
        exps = np.exp(probs)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.value = np.sum(-np.eye(self.n_classes)[targets] * np.log(probs))
        return self

        # End of todo

    def backward(self):
        # TODO Implement backward propogation
        # of softmax loss function.

        return self.probs - np.eye(self.n_classes)[self.targets]
        ...

        # End of todo

class CrossEntropyLoss(Loss):
    def LOSS(self, probs, targets):
        self.nx = probs
        self.ny = targets
        shifted_x = self.nx - np.max(self.nx)
        ex = np.exp(shifted_x)
        sum_ex = np.sum(ex)
        self.softmax = ex / sum_ex
        self.ny=self.ny.reshape((1000,1))
        self.entropy = - np.log(self.softmax) * self.ny

        self.loss = np.sum(self.entropy)
        return self.loss

    def backward(self):
        self.dnx = self.softmax.copy() * np.sum(self.ny)
        self.dnx -= self.ny
        return self.dnx

        """"# TODO Calculate cross-entropy loss.

        #super(SoftmaxLoss,self).__call__(probs, targets)
        self.nx=probs
        self.ny=targets
        for i in range (1000):
         at = targets[i]
         loss=np.sum(np.log(probs[i][at]))
        ...
        return loss

        # End of todo

    def backward(self,probs,targets):
        # TODO Implement backward propogation
        # of cross-entropy loss function.
        targets=targets.reshape((1000,1))
        self.dnx = - targets / probs
        return self.dnx

        # End of todo"""
"""class CrossEntropyLoss(Loss):
    def __init__(self):
      self.nx = None
      self.ny = None
      self.dnx = None

    def loss(self, nx, ny):
      self.nx = nx
      self.ny = ny
      loss = np.sum(- ny * np.log(nx))
      return loss
    def backward(self):
      self.dnx = - self.ny / self.nx
      return self.dnx"""



