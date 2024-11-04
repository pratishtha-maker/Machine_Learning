# Library
import sys
import numpy as np
from sklearn.preprocessing import normalize

# standard perceptron algorithm
def _std_PR(lr = 1.0):
    # max epoch
    max_epoch = 10
    # weights initialization
    weights = np.zeros((1,4), dtype = "float")
    weights = np.asarray(weights)
    for epoch in range(max_epoch):
        weights = _std_epoch(weights,lr)      
        tr_pred = np.sign(weights* np.transpose(X_tr))
        # calculate training error
        tr_error = _calculate_error(tr_pred, Y_tr)
        ts_pred = np.sign(weights* np.transpose(X_ts))
        # calculate test error
        ts_error = _calculate_error(ts_pred, Y_ts)
        # results
        print("Epoch:", epoch+1 , "Training Error:" , tr_error, "Test Error:", ts_error)
    print ("Final weight:" ,weights)
    
    
# standard epoch

def _std_epoch(weights,lr):
    [X, Y] = _shuffle_examples(X_tr,Y_tr)
    for i in range(Y.size):
        [X_i,Y_i] = _get_example(i,X,Y)
        if _pred_error(weights,X_i,Y_i):
            weights = _new_weights(weights,X_i,Y_i,lr)
    return np.any(weights)

# shuffle example
def _shuffle_examples(X,Y):
    #concate
    append_xy = np.c_[X.reshape(len(X),-1),Y.reshape(len(Y),-1)]
    np.random.shuffle(append_xy)
    shfl_X = append_xy[:,:X.size // len(X)].reshape(X.shape)
    shfl_Y = append_xy[:,X.size // len(X):].reshape(Y.shape)
    return [shfl_X,shfl_Y]

def _voted_PR(lr=1.0):
    # max epoch
    max_epoch = 10
    # weights initialization
    weights = np.zeros((1,4), dtype = "float")
    weights = np.asarray(weights)
    m = 0
    votes = [0.0]
    for epoch in range(max_epoch):
        [weights, votes, m] = _voted_epoch(weights,votes,lr,m)
        
        tr_pred = _calculate_voted_pred(weights, votes, X_tr, len(Y_tr))
        # calculate training error
        tr_error = _calculate_error(tr_pred, Y_tr)
        
        ts_pred = _calculate_voted_pred(weights, votes, X_ts, len(Y_ts))
        # calculate test error
        ts_error = _calculate_error(ts_pred, Y_ts)
        # results
        print("Every epoch:", epoch+1 , "Training Error:" , tr_error, "Test Error:", ts_error)
    print("Vote:", votes)
    print ("Final weight:" ,weights)
    
def _voted_epoch(weights,votes,lr,m):
    
    for i in range(Y_tr.size):
        [X_i, Y_i] = _get_example(i,X_tr,Y_tr)
        
        if _pred_error(weights[m],X_i,Y_i):
            wt_m = _new_weights(weights[m],X_i,Y_i,lr)
            weights = np.r_[weights, wt_m]
            votes.append(1)
            m += 1
        else:
            votes[m] += 1
    return np.any[weights, votes, m]

def _calculate_voted_pred(weights,votes, X, Y_len):
    vote_pred = np.zeros((1,Y_len), dtype ="float" )
    for i in range(len(votes)):
        predict = np.sign(weights[i] * np.transpose(X))
        vote_pred += votes[i] * predict
    return np.sign(vote_pred)
        
    
def _avg_PR(lr =1.0):
    # max epoch
    max_epoch = 10
    # weights initialization
    weights = np.zeros((1,4), dtype = "float")
    weights = np.asarray(weights)
    average = weights.copy()
   
    for epoch in range(max_epoch):
        [average, weights] = _avg_epoch(weights, average, lr)
        
        tr_pred = np.sign(average * np.transpose(X_tr))
        # calculate training error
        tr_error = _calculate_error(tr_pred, Y_tr)
        
        ts_pred = np.sign(weights * np.transpose(X_ts))
        # calculate test error
        ts_error = _calculate_error(ts_pred, Y_ts)
        # results
        print("Every epoch:", epoch+1 , "Training Error:" , tr_error, "Test Error:", ts_error)
    print("Average:", average)
    print ("Final weight:" ,weights)

def _avg_epoch(weights, average, lr):
    for i in range(Y_tr.size):
        [X_i, Y_i] = _get_example(i,X_tr,Y_tr)
        
        if _pred_error(weights,X_i,Y_i):
          
            weights = _new_weights(weights,X_i,Y_i,lr)
        
        average = average + weights

    return np.any[average, weights] 

def _get_example(ex_index,X,Y):
    X_i = np.zeros((1, 4),dtype ="float")
    for attr in range(4):
        X_i[0, attr] = X[ex_index, attr]
    return [X_i, Y[ex_index]]

def _pred_error(weights,X_i,Y_i):
    pred = np.sign(X_i* np.transpose(weights))
    return np.any(Y_i !=pred)

def _new_weights(weights,X_i,Y_i,lr):
    return np.any(weights+ lr * Y_i * X_i)


def _calculate_error(pred,Y):
   
    return np.count_nonzero(np.multiply(np.transpose(Y),pred)== -1)/len(Y)

# import data from main 
def _dataloader(path,num_of_examples):
    # load data from input path with number of examples
    X = np.zeros((num_of_examples, 4),dtype ="float")
    Y = np.zeros((num_of_examples, 1),dtype ="float")
    with open(path,'r') as f:
        i = 0
        for line in f:
            example = []
            terms = line.strip().split(',')
            for j in range(len(terms)):
                if j == 4:
                    # target transformation(0,0) to (-1,1)
                    Y[i] = 2* float(terms[j]) -1
                else:
                    example.append(float(terms[j]))
            X[i] = example
            i = i+1
        X = normalize(np.asarray (X),axis= 0)
        return [np.asarray(X),np.asarray(Y)]
        
        
if __name__ == '__main__':
    [X_tr, Y_tr] = _dataloader("/Users/pratishthaagnihotri/Documents/Machine_Learning/HW_3/bank-note/train.csv", 872)
    [X_ts, Y_ts] = _dataloader("/Users/pratishthaagnihotri/Documents/Machine_Learning/HW_3/bank-note/test.csv", 500)
    if sys.argv[1] == "standard":
        _std_PR()
    elif sys.argv[1] == "voted":
        _voted_PR()
    elif sys.argv[1] == "average":
        _avg_PR()