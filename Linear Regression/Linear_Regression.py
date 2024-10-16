
import sys
import numpy as np
import math
import random
from numpy import linalg as lin
import matplotlib.pyplot as plt



def cost_func(x, y, weights):
    """Computes the Least Mean Squares cost."""
    sum_cost = 0.0
    
    for ex in range(len(x)):
        cost = y[ex] - np.dot (weights,x[ex])
        sum_cost += np.square(cost)
    return float(0.5 * sum_cost)


# Batch Gradient Descent
def run_bgd(x,y,lr):
    """
    Runs Batch Gradient Descent for Least Mean Squares algorithm.
    The algorithm is said to converge 
    when ||new weight vector - prev weight vector||< 10e-6.
    The learning rate is halved every iterations and tune rate 
    """
    costs = []
    weights = np.zeros(x.shape[1])
    diff = math.inf
    while diff > 10e-6:
        grad_w = np.zeros(x.shape[1])
        for j in range(len(x[0])):
            grad_wj = 0
            for i in range(len(x)):
                grad_wj += x[i][j] * (y[i] - np.dot(weights, x[i]))
            grad_w[j] = grad_wj
        weights_T = weights + lr * grad_w
        diff = lin.norm(weights - weights_T)
        costs.append(cost_func(x,y, weights))
        weights = weights_T
    costs.append(cost_func(x,y, weights))
    return weights, costs

# Stochastic Gradient Descent
def run_sgd(x,y,lr):
    """
    Runs stochastic gradient descent for Least Mean Squares algorithm.
    The algorithm is said to converge when 
    the change in the current cost from the previous cost is<10e-10
    The learning rate is halved every 100 iterations
     """
    weights = np.zeros(x.shape[1])
    diff = math.inf
    costs = [cost_func(x,y,weights)]
    while diff > 10e-10:
        i = random.randrange(len(x))
        grad_w = np.zeros(x.shape[1])
        for j in range(len(x[0])):
            grad_w[j] = x[i][j] *(y[i] - np.dot(weights, x[i]))
        weights_T = weights + lr * grad_w
        weights = weights_T
        new_cost = cost_func(x,y, weights)
        diff = abs(new_cost - costs[-1])
        costs.append(new_cost)
    return weights,costs
      
def run_optimal(x,y):
    x_T = x.T
    xx = np.matmul(x_T,x_T.T)
    inv_xx = lin.inv(xx)
    weights_new = np.matmul(np.matmul(inv_xx, x_T), y)
    return weights_new
    
        

        
if __name__ == '__main__':
    """load data from path to a csv file
    with the given amount of examples."""
    train = np.loadtxt("/Users/pratishthaagnihotri/Documents/learn/ML/ML_HW2/concrete/train.csv", delimiter =',',usecols = range(8))
    #test data    
    test = np.loadtxt("/Users/pratishthaagnihotri/Documents/learn/ML/ML_HW2/concrete/train.csv", delimiter =',',usecols = range(8))

    # get vector x and y for both train and test datasets
    X_train = train[:,:-1]
    one_train = np.ones(X_train.shape[0])
    D_train = np.column_stack((one_train, X_train))
    Y_train = train[:,-1]

    X_test = test[:,:-1]
    one_test = np.ones(X_test.shape[0])
    D_test = np.column_stack((one_test, X_test))
    Y_test = test[:,-1]
    
   
    if sys.argv[1] == "bgd":
        lr = 0.002
        w, costs = run_bgd(D_train, Y_train, lr)
        print("learned weight: ", w)
        print("learned rates: ", lr)
        print("Test data evaluation Cost: ", cost_func(D_test, Y_test, w))
        fig1 = plt.figure()
        plt.plot(costs)
        fig1.suptitle('Gradient Descent ', fontsize=15)
        plt.xlabel('Iteration', fontsize=15)
        plt.ylabel('Cost Function Value (CFV)', fontsize=15)
        plt.show()
        fig1.savefig("BGD_cost_function.png")
        print("Figure has been saved!")
    if sys.argv[1] == "sgd":
        lr = 0.001
        w, costs = run_sgd(D_train, Y_train, lr)
        print("learned weight: ", w)
        print("learned rates: ", lr)
        print("Test data evaluation Cost: ", cost_func(D_test, Y_test, w))
        fig2 = plt.figure()
        plt.plot(costs)
        fig2.suptitle('Gradient Descent ', fontsize=15)
        plt.xlabel('Iteration', fontsize=10)
        plt.ylabel('Cost Function Value (CFV)', fontsize=10)
        plt.show()
        fig2.savefig("SGD_cost_function.png")
        print("Figure has been saved!")
    if sys.argv[1] == "optimal":
    
        w_optimal = run_optimal(D_train, Y_train)
        print("learned weight: ",w_optimal)

        print("Test data evaluation Cost: ", cost_func(D_test, Y_test, w_optimal))

