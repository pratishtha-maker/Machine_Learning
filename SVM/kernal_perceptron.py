import sys
import pandas as pd
import numpy as np
from numpy import linalg

# Kernel Perceptron
def run_KP():

    train = pd.read_csv("./bank-note/train.csv", header=None)
    test = pd.read_csv("./bank-note/test.csv", header=None)

    train_x = train.iloc[: , :-1]
    train_x = np.array(train_x)
    train_y = train.iloc[: , -1]
    train_y = np.where(train_y > 0, 1, -1)
    train_y = np.array(train_y)

    test_x = test.iloc[: , :-1]
    test_x = np.array(test_x)
    temp_y = test.iloc[: , -1]
    test_y = np.where(temp_y > 0, 1, -1)
    test_y = np.array(test_y)
    # No shuffle
    gamma = [0.1, 0.5, 1, 5, 100]
    for g in gamma:
        kp = KernelPerceptron(kernel="gaussian", gamma=g)
        kp.fit(train_x, train_y)
        pred_train = kp.predict(train_x)
        pred_test = kp.predict(test_x)
        corr_train = np.sum(pred_train == train_y)
        corr_test = np.sum(pred_test == test_y)
        print("learning rate,gamma " , str(g), "Train Accuracy " , str(corr_train / len(pred_train)), "Test Accuracy " , str(corr_test / len(pred_test)))
        print("_________")


class KernelPerceptron():
    def __init__(self, kernel, gamma, T=1):
        #intialize parameters
        self.kernel = kernel
        self.T = T
        self.gamma = gamma
        self.alpha = 5
        
    def predict(self, X):
        # Computes prediction
        X = np.atleast_2d(X)
        return np.sign(self.dual_objective(X))

    def linear(self, xi, xj):
        # Computes linear kernel
        return np.dot(xi, xj)

    def gaussian(self, X, z, gamma):
        # Computes gaussian kernel
        return np.exp(-(np.linalg.norm(X-z, ord=2)**2) / gamma)
    
    def dual_objective(self, X):
        # Computes objective function
        pred_y = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, support_vectors in zip(self.alpha, self.sv_y, self.support_vectors):
                s += a * sv_y * self.gaussian(X[i], support_vectors, self.gamma)
            pred_y[i] = s
        return pred_y    
    
    def fit(self, X, y):
        # Compute convergence: from svm function
        threshold = 1e-10
        self.alpha = np.zeros(X.shape[0], dtype=np.float64)

        K = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if self.kernel == "gaussian":
                    K[i,j] = self.gaussian(X[i], X[j], self.gamma)
                elif self.kernel == "linear":
                    K[i,j] = self.linear(X[i], X[j])

        for t in range(self.T):
            for i in range(X.shape[0]):
                if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
                    self.alpha[i] += 1.0

        support_vectors = self.alpha > threshold
        ind = np.arange(len(self.alpha))[support_vectors]
        self.alpha = self.alpha[support_vectors]
        self.support_vectors = X[support_vectors]
        self.sv_y = y[support_vectors]





if __name__ == "__main__":
    run_KP()