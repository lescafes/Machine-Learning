import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time

def load_mnist(mode='train', n_samples=None, flatten=True):
    images = r'F:/pycharm code/train-images-idx3-ubyte' if mode == 'train' else 'F:/pycharm code/t10k-images-idx3-ubyte'
    labels = r'F:/pycharm code/train-labels-idx1-ubyte' if mode == 'train' else 'F:/pycharm code/t10k-labels-idx1-ubyte'
    length = 60000 if mode == 'train' else 10000
    X = np.fromfile(open(images), np.uint8)[16:].reshape((length, 28, 28)).astype(np.int32)
    if flatten:
        X = X.reshape(length, -1)
    y = np.fromfile(open(labels), np.uint8)[8:].reshape(
        (length)).astype(np.int32)
    return (X[:n_samples] if n_samples is not None else X,
            y[:n_samples] if n_samples is not None else y)


class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    num_test = X.shape[0]
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
    cook=np.zeros(shape=(10000,4),dtype = self.ytr.dtype)##7个最近邻的标签矩阵
    # loop over all test rows
    for i in range(num_test):

       distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
       for j in range(4):
         min_index = np.argmin(distances) # get the index with smallest distance
         cook[i][j] = self.ytr[min_index] # predict the label of the nearest example
         distances= np.delete(distances,[min_index])

    for k in range(10000):
       abs=np.bincount(cook[k])##统计次数的投票矩阵
       Ypred[k]=np.argmax (abs,axis=0)

    return Ypred

Xtr, Ytr = load_mnist('train')
Xte, Yte = load_mnist('test')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 28 * 28 )
Xte_rows = Xte.reshape(Xte.shape[0], 28 * 28 )

nn = NearestNeighbor()
nn.train(Xtr_rows, Ytr)
Yte_predict = nn.predict(Xte_rows)
print ('accuracy: %f' % ( np.mean(Yte_predict == Yte) ))

