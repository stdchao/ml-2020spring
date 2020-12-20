import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionClassifier:
  def __init__(self, dims, learning_rate=0.1):
    self.w = np.zeros((dims,))
    self.b = np.zeros((1,))
    self.learning_rate = learning_rate
  
  def sigmoid(self, z):
    return np.clip(1.0/(1.0 + np.exp(-z)), 1e-8, 1-(1e-8))

  def forward(self, X):
    return self.sigmoid(np.matmul(X, self.w) + self.b)

  def accuracy(self, y_pred, y_label):
    return 1 - np.mean(np.abs(np.round(y_pred) - y_label))
  
  def corss_entropy_loss(self, y_pred, y_label):
    return - np.dot(y_label, np.log(y_pred)) - np.dot(1-y_label, np.log(1-y_pred))

  def update_gradient(self, X, y_label, step=1):
    y_pred = self.forward(X)
    y_error = y_label - y_pred
    w_grad = -np.sum(y_error * X.T, 1)
    b_grad = -np.sum(y_error)
    self.w -= self.learning_rate/np.sqrt(step) * w_grad
    self.b -= self.learning_rate/np.sqrt(step) * b_grad

def shuffle(X, Y):
  randomsize = np.arange(X.shape[0])
  np.random.shuffle(randomsize)
  return X[randomsize], Y[randomsize]

if __name__ == "__main__":
  # parser args
  parser = argparse.ArgumentParser()
  parser.add_argument("X_train_path", metavar="X_train", help="X_train")
  parser.add_argument("Y_train_path", metavar="Y_train", help="Y_train")
  parser.add_argument("X_test_path", metavar="X_test", help="X_test")
  parser.add_argument("output_file", metavar="Output_test", help="Output")
  args = parser.parse_args()

  # load data
  with open(args.X_train_path, 'r') as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype='float')
  with open(args.Y_train_path, 'r') as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype='float')
  with open(args.X_test_path, 'r') as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype='float')
  print(X_train.shape, Y_train.shape, X_test.shape)
  
  # nomalize features
  X_mean = np.mean(X_train, axis=0).reshape(1, -1)
  X_std = np.std(X_train, axis=0).reshape(1, -1)
  X_train = (X_train - X_mean) / (X_std + 1e-8)
  X_test = (X_test - X_mean) / (X_std + 1e-8)
  print(X_train.shape, X_test.shape, X_mean.shape, X_std.shape)

  # split dev
  train_size = int(0.9 * X_train.shape[0])
  X_dev = X_train[train_size:]
  Y_dev= Y_train[train_size:]
  X_train = X_train[:train_size]
  Y_train = Y_train[:train_size]
  print('train {}; dev {}; test {}; dim {}'.format(
    X_train.shape[0], X_dev.shape[0], X_test.shape[0], X_train.shape[1]))

  # build model
  size, dims = X_train.shape
  lr_clf = LogisticRegressionClassifier(dims)

  # train model
  ## hyperparameters
  max_iter = 10
  batch_size = 32
  learning_rate = 0.2

  ## loss and acc
  train_loss, dev_loss, train_acc, dev_acc = [], [], [], []

  ## loop epoch
  step = 1
  for epoch in range(max_iter):
    X_train, Y_train = shuffle(X_train, Y_train)

    ### loop minibatch
    for idx in range(int(size/ batch_size)+1):
      X = X_train[idx*batch_size:(idx+1)*batch_size]
      Y = Y_train[idx*batch_size:(idx+1)*batch_size]
    
      #### update gradient with decay learning rate
      lr_clf.update_gradient(X, Y, step)
      step += 1
    
    ### compute loss and acc
    train_pred = lr_clf.forward(X_train)
    train_loss.append(lr_clf.corss_entropy_loss(train_pred, Y_train) / size)
    train_acc.append(lr_clf.accuracy(train_pred, Y_train))

    dev_pred = lr_clf.forward(X_dev)
    dev_loss.append(lr_clf.corss_entropy_loss(dev_pred, Y_dev) / X_dev.shape[0])
    dev_acc.append(lr_clf.accuracy(dev_pred, Y_dev))
  
  print('train loss {} and acc {}; dev loss {} and acc {}'.format(
    train_loss[-1], train_acc[-1], dev_loss[-1], dev_acc[-1]
  ))

  # plot loss and acc curve during training
  fig, axes = plt.subplots(1, 2, figsize=(12, 6))
  axes[0].plot(train_loss)
  axes[0].plot(dev_loss)
  axes[0].set_title('Loss')
  axes[0].legend(['train', 'dev'])
  axes[1].plot(train_acc)
  axes[1].plot(dev_acc)
  axes[1].set_title('Acc')
  axes[1].legend(['train', 'dev'])
  plt.savefig('train_loss_acc.png')
  
  # print most significant weights
  index = np.argsort(np.abs(lr_clf.w))[::-1]
  with open(args.X_test_path, 'r') as f:
    content = f.readline().strip('\n').split(',')
    features = np.array(content)
  for i in index[0:10]:
    print(features[i], lr_clf.w[i])

  # predict test
  test_pred = np.round(lr_clf.forward(X_test)).astype(np.int)
  with open(args.output_file, 'w') as f:
    f.write('id,label\n')
    for i,label in enumerate(test_pred):
      f.write('{},{}\n'.format(i, label))