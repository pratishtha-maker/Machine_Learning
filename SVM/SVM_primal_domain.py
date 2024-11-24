import sys
import numpy as np
from scipy.optimize import minimize

# Primal
def primal_svm(sch=1):
    # Calculates the hyperparameters, then computes the train and test errors for every value of c.
    cs = [ 100 / 873, 500 / 873, 700 / 873]
    train_errors = []
    test_errors = []

    hyperparameters = get_primal_hyperparams(sch)
    gamma = hyperparameters[0]
    a = hyperparameters[1]

    #print("GAMMA: ", gamma)
    if sch == 1: print("a: ", a)

    for c in cs:
        [train_error, test_error] = get_primal_all_errors(c, gamma, a)
        train_errors.append([c, train_error])
        test_errors.append([c, test_error])

    for i in range(len(train_errors)):
        print("C:", cs[i], " Train Error:", train_errors[i][1], " Test Error:", test_errors[i][1])


def get_primal_all_errors(c, gamma, a):
    # Calculates the weights by running SGD, then computes train and test errors for the given parameters
    [weights, _] = run_sgd(c, gamma, a)
    weights = weights[0, :weights.size - 1]

    train_error = get_primal_err(weights, train_y, train_data)
    test_error = get_primal_err(weights, test_y, test_data)
    return [train_error, test_error]


def get_primal_hyperparams(sch):
    # Computes the best hyper parameters for c = 700 / 873
    gammas = [0.01, 0.005, 0.0025, 0.00125, 0.000625]
    a_s = [1, 0.1, 0.05, 0.01, 0.001]
    smallest = [0, 100.0, 100.0]
    c = 700 / 873

    for gamma in gammas:
        if sch == 1:
            for a in a_s: smallest = get_smallest_error(smallest, c, gamma, a)
        else:
            smallest = get_smallest_error(smallest, c, gamma)

    return smallest


def get_smallest_error(smallest, c, gamma, a=0):
    # Computes the error for the given parameters and returns the error if it is the smallest, or the previous smallest.
    [weights, _] = run_sgd(c, gamma, a)
    weights = weights[0, :weights.size - 1]
    error = get_primal_err(weights, train_y, train_data)

    if error < smallest[2]: smallest = [gamma, a, error]
    return smallest


def get_primal_err(weights, y, x):
    # Computes the predictions for the given weights/data. Then calculates the error for the predictions.
    predictions = get_primal_pred(weights, x)
    return get_err(y, predictions)


def get_primal_pred(w, x):
    return np.sign(w * np.transpose(x))


# SGD
def run_sgd(c, gamma_0, a, n=872):
    weights = np.array([[0., 0., 0., 0., 0.]])
    #weights = np.array([[0., 0., 0., 0.]])
    loss = []

    for epoch in range(0, 100):
        # Set learning rate
        gamma = update_gamma(epoch, gamma_0, a)
        [tr_ex_y, tr_ex_x] = _shuffle_data(train_y, train_data)

        for i in range(0, n):
            # Augment with bias parameter
            x = np.append(tr_ex_x[i], [[1]], axis=1)

            if is_incorrect(tr_ex_y[i], weights, x):
                weights = update_incorrect_example(weights, gamma, c, tr_ex_y[i], x, n)
            else:
                weights[0][:weights.size-1] = update_correct_example(weights, gamma)

            # To make sure converging
            l = compute_loss(c, tr_ex_y, weights, tr_ex_x, n)
            loss.append(l[0, 0])
    #print("LOSS: ", loss)
    return [weights, loss]


def update_gamma(t, gamma_0, a):
    if a != 0: return gamma_0 / (1.0 + (gamma_0 / a) * t)
    else: return gamma_0 / (1 + t)


def _shuffle_data(y, data):
    #Shuffles the given data by appending y to the data
    combined = np.c_[data.reshape(len(data), -1), y.reshape(len(y), -1)]
    np.random.shuffle(combined)
    shuffled_data = combined[:, :data.size // len(data)].reshape(data.shape)
    shuffled_y = combined[:, data.size // len(data):].reshape(y.shape)
    return [shuffled_y, shuffled_data]


def is_incorrect(y_i, w, x_i):
    return y_i * np.dot(x_i, np.transpose(w)) <= 1


def update_incorrect_example(w, gamma, c, y_i, x_i, n=872):
    w[0, w.size - 1] = 0
    temp = (1-gamma) * w + gamma * c * n * y_i * x_i
    return temp[0]


def update_correct_example(w, gamma):
    return (1-gamma) * w[:w.size - 1]


def compute_loss(c, y, w, x, n=872):
    x = np.append(x, np.ones((n, 1)), axis=1)
    hinge = max(0, 1 - np.transpose(y) * np.dot(x, np.transpose(w)))

    w = w[0, :w.size - 1]
    regularization = 1/2 * w.dot(w.T)
    return regularization + c * hinge

def get_err(y, predictions):
    return 1 - np.count_nonzero(np.multiply(y.T, predictions) == 1) / len(y)

# Import
def data_loader(path, num_examples):
    #Imports the data at the given path to a csv file with the given amount of examples
    data = np.empty((num_examples, 4), dtype="float")
    y = np.empty((num_examples, 1), dtype="float")

    with open(path, 'r') as f:
        i = 0
        for line in f:
            example = []
            terms = line.strip().split(',')
            for j in range(len(terms)):
                if j == 4:
                    y[i] = 2 * float(terms[j]) - 1
                else:
                    example.append(float(terms[j]))
            data[i] = example
            i += 1
    return [np.asmatrix(data), np.asmatrix(y)]





if __name__ == '__main__':
    [train_data, train_y] = data_loader("./bank-note/train.csv", 872)
    [test_data, test_y] = data_loader("./bank-note/test.csv", 500)

    if sys.argv[1] == "primal_gamma1":
        primal_svm()
    elif sys.argv[1] == "primal_gamma2":
        primal_svm(2)
