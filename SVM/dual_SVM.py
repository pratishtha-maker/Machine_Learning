import pandas as pd
import numpy as np
from scipy.optimize import minimize
def run_dual():
    #args = sys.argv[1:]
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

    gamma_0 = 1
    alpha = 1

    gamma1 = lambda t : gamma_0 / (1 + (gamma_0/alpha)*t)
    gamma2 = lambda t : gamma_0 / (1 + t)
    Cs = [100/873, 500/873, 700/873]
    gamma = [0.1, 0.5, 1.0, 5.0, 100.0]    
#ques 3a    
   
    print("Dual SVM - Linear")
   
    for c in Cs:
        svm = DualSVM(train_x, train_y, C=c, kernel="linear", gamma=None)
        svm.fit(train_x, train_y)
        print("C : " ,c )
        print("Learned Weights: ", svm.wstar)
        print("Learned Bias: " ,svm.bstar) 
        print("SVM Counts: ",len(svm.support_vectors))
        print("Training Accuracy: %.3f" % svm.get_accuracy(train_x, train_y))
        print("Testing Accuracy: %.3f" % svm.get_accuracy(test_x, test_y) )
        print("--------------")
#ques 3b

    svs = []
    
    print("Dual SVM - Gaussian Kernel ")
    
    for c in Cs:
        for gs in gamma:
            svm = DualSVM(train_x, train_y, C=c, kernel="gaussian", gamma=gs)
            svm.fit(train_x, train_y)
            print("C : " ,c ," and gamma: ",gs)
            print("Learned Weights: " ,svm.wstar)
            print("Learned Bias: " ,svm.bstar)
            print("SVM Counts: " ,len(svm.support_vectors)) 
            print("Training Accuracy: %.3f" % svm.get_accuracy(train_x, train_y) )
            print("Testing Accuracy: %.3f" % svm.get_accuracy(test_x, test_y) )
            print("--------------")
            if c == 500/873:
                svs.append(svm.support_vectors)
#ques 3c
    for i in range(len(svs)):
        count = 0
        for v in np.array(svs[i]):
            if v in np.array(svs[i+1]):
                count += 1

        print("Overlapping vector count " + str(gamma[i]) + " to " + str(gamma[i+1]) + ": " + str(count))
        print("--------------")
        
class DualSVM:
    def __init__(self, X, y, C, kernel = "dot", gamma=0.1):
        self.wstar = np.ndarray
        self.bstar = float
        self.C = C
        self.gamma = gamma
        self.support_vectors = []
        self.kernel = kernel

    def fit(self, X, y):
        cons = [
            {
                'type': 'eq',
                'fun': lambda a: np.sum(a*y)
            },
            {
                'type': 'ineq',
                'fun': lambda a : a 
            },
            {
                'type': 'ineq',
                'fun': lambda a: self.C - a 
            }
        ]
        
        out = minimize(self.dual_objective, x0=np.zeros(shape=(len(X),)), args=(X, y), method='SLSQP', constraints=cons)

        self.support_vectors = np.where(0 < out.x)[0]  

        self.wstar = np.zeros_like(X[0])
        for i in range(len(X)):
            self.wstar += out['x'][i]*y[i]*X[i]

        self.bstar = 0
        for j in range(len(X)):
            self.bstar += y[j] - np.dot(self.wstar, X[j])
        self.bstar /= len(X)
                  
    def get_accuracy(self, X, y):
        corr = 0
        length = len(y)
        preds = self.predict(X)

        for i in range(length):
            if preds[i] == y[i]:
                corr += 1
        return corr/length
    def linear(self, X):
        return (X@X.T)

    def gaussian(self, X, z, gamma):
        return exp(-(np.linalg.norm(x-z, ord=2)**2) / gamma)

    def dual_objective(self, a, X, y):
        yy_mat = y * np.ones((len(y), len(y)))
        alpha_mat = a * np.ones((len(a), len(a)))

        if self.kernel == 'linear':
            xj = self.linear(X)
        if self.kernel == 'gaussian':
            xj = X**2 @ np.ones_like(X.T) - 2*X@X.T + np.ones_like(X) @ X.T**2 
            xj = np.exp(-( xj / self.gamma))

        vals = (yy_mat*yy_mat.T) * (alpha_mat*alpha_mat.T) * xj
        return 0.5*np.sum(vals) - np.sum(a)



    def predict(self, X, kernel = "linear") -> np.ndarray:
        if kernel == 'linear':
            pred = lambda d : np.sign(np.dot(self.wstar, d) + self.bstar)
        if kernel == 'gaussian':
            pred = lambda d : np.sign(self.gaussian(self.wstar, d, self.gamma) + self.bstar)
        return np.array([pred(xi) for xi in X])



if __name__ == "__main__":
    run_dual()