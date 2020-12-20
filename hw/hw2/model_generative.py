import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument('X_train_path', metavar='X_train', help='X_train')
  parser.add_argument('Y_train_path', metavar='Y_train', help='Y_train')
  parser.add_argument('X_test_path', metavar='X_test', help='X_test')
  parser.add_argument('output_file', metavar='output', help='output')
  args = parser.parse_args()

  # load data
  with open(args.X_train_path) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype='float')
  with open(args.Y_train_path) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype='float')
  with open(args.X_test_path) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype='float')
  dims = X_train.shape[1]

  # normalize features
  X_mean = np.mean(X_train, axis=0).reshape(1, -1)
  X_std = np.std(X_train, axis=0).reshape(1, -1)
  X_train = (X_train - X_mean) / (X_std + 1e-8)
  X_test = (X_test - X_mean) / (X_std + 1e-8)

  # compute mean and covariance by class
  X_train_0 = X_train[np.where(Y_train == 0)]
  X_train_1 = X_train[np.where(Y_train == 1)]
  mean_0 = np.mean(X_train_0, axis=0)
  mean_1 = np.mean(X_train_1, axis=0)

  cov_0 = np.zeros((dims, dims))
  cov_1 = np.zeros((dims, dims))
  for x in X_train_0:
    cov_0 = np.dot(np.transpose(x - mean_0), x - mean_0) / X_train_0.shape[0]
  for x in X_train_1:
    cov_1 = np.dot(np.transpose(x - mean_1), x - mean_1) / X_train_1.shape[0]
  cov_share = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / \
    (X_train_0.shape[0] + X_train_1.shape[1])
  
  # compute weights and bias
  
