"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        n_data, dimension = X_train.shape
        grads_w = np.zeros((dimension, self.n_class))        
        scores = X_train.dot(self.w) #(N,C)
        scores = scores - np.max(scores, axis = 1).reshape(-1,1)
        exp_scores = np.exp(scores) #
        #print(exp_scores)(N,C)
        #print(np.sum(exp_scores, axis=1, keepdims=True)) -> N,1
        prob_scores =  exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        dscore = prob_scores
        dscore[np.arange(n_data), y_train] -= 1        
        grads_w = np.dot(X_train.T, dscore)
        grads_w /= n_data
        grads_w += self.reg_const * self.w
        return grads_w
        
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        n_data, dimension = X_train.shape
        #print(n_data, dimension)
        self.w = np.random.rand(dimension, self.n_class)
        batch_size = 1
        for i in range(self.epochs):
            indices = np.random.choice(n_data, batch_size)
            X_batch = X_train[indices]
            y_batch = y_train[indices]
            grads_w = self.calc_gradient(X_batch, y_batch)
            self.w -= self.lr * grads_w
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.argmax(X_test.dot(self.w), axis = 1)
