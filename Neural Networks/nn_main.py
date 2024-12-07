import numpy as np
import nn_layer

import pandas as pd

def main():
    
    train = pd.read_csv('/Users/pratishthaagnihotri/Documents/Machine_Learning/HW_5/bank-note/train.csv', header=None)
    test = pd.read_csv('/Users/pratishthaagnihotri/Documents/Machine_Learning/HW_5/bank-note/test.csv', header=None)

    Xtr = np.array(train.iloc[: , :-1])
    ytr = np.array(train.iloc[: , -1])
    Xte = np.array(test.iloc[: , :-1])
    yte = np.array(test.iloc[: , -1])



    #weights = [5, 10, 25, 50, 100]
    weights = [0, 0, 0, 0, 0]
    gammas = [0.5, 0.5, 0.05, 0.1, 0.01]
    gamma_idx = 0
    for weight in weights:
        yte = np.array(yte)

        net = NeuralNetwork(nn_layer.create_layers(3, 4, weight, 1))
        print("========================================")
        print("3 Layered NN Model - Width = " + str(weight))
        training_acc = net.fit(net, Xtr, ytr, lr_0 = gammas[gamma_idx], d = 1)
        print("Train error: " + str(sum(training_acc)/len(training_acc)))
        testing_acc = net.get_acc(net, Xte, yte)
        gamma_idx += 1


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.epochs = 50

    def fit(self, net, X, y, lr_0 = 0.5, d = 1):
        iter_losses = []

        for e in range(self.epochs):
            losses = []
            idxs = np.arange(len(X))

            # Reshuffle the data at the beginning of each epoch
            np.random.shuffle(idxs)
            for i in idxs:
                y_pred, zs = self.forward(X[i])
                losses.append(self.square_loss(y_pred, y[i]))
                #lr = lr_0/2
                # Update learning rate by this learning function
                lr = lr_0 / (1 + (lr_0/d)*e)
                self.backward(zs, y[i], lr)
            iter_losses.append(np.mean(losses))

        return iter_losses

    def get_acc(self, net, X, y):
        losses = []
        for i in range(len(X)):
            y_pred, _ = self.forward(X[i])
            losses.append(self.square_loss(y_pred, y[i]))
        print("Test error:" + str(np.mean(losses)))

        return np.mean(losses)

    def forward(self, x): 
        x = np.append(1, x)
        zs = [np.atleast_2d(x)]

        for l in range(len(self.layers)):
            out = self.layers[l].evaluation(zs[l])
            zs.append(out)

        return np.float64(zs[-1]), zs
        #return (zs[-1]), zs

    def backward(self, zs, y, lr = 0.1):

        partials = [zs[-1] - y]

        for l in range(len(zs) - 2, 0, -1):
            delta = self.layers[l].backprop(zs[l], partials)
            partials.append(delta)
    
        partials = partials[::-1]

        for l in range(len(self.layers)):
            grad = self.layers[l].update_ws(lr, zs[l], partials[l])


    def square_loss(self, pred, target):
        return 0.5*(pred - target)**2

if __name__ == "__main__":
    main()