
import math
import random
import sys
sys.path.append('../DecisionTree')
import numpy as np
import ID3


#variables
T = 500
m = 4999
trees = []
vote_arr = []
attr_subset_nums = [2, 4, 6]
attr_subset_num = 0


# =============================================================================
# #Function to implement Ada boost

# ============================================================================= 
def _run_ada_boost():
    """Runs T iterations of the ID3 algorithm on Decision Stumps"""
    global predictions
    test_predictions = np.zeros((T, m))
    y_train = np.array(ID3.train_data[-1])
    y_test = np.array(ID3.test_data[-1])
    train_err_str = ""
    test_err_str = ""

    for t in range(T):
        arr = _train_iter_ada_boost(t, y_train)
        train_err_str += str(_get_PE(y_train, arr[0])) + ","

        test_hyp = _test_iter_ada_boost(arr[1], test_predictions)
        test_err_str += str(_get_PE(y_test, test_hyp)) + ","

    print("TRAIN: ")
    print(train_err_str)
    print("TEST: ")
    print(test_err_str)
    

    train_err_str = ""
    test_err_str = ""
    for t in range(T):
        train_err_str += str(ID3.predict(ID3.train_data, trees[t].root)) + ","
        test_err_str += str(ID3.predict(ID3.test_data, trees[t].root)) + ","

    print("DEC STUMP TRAIN: ")
    print(train_err_str)
    print("DEC STUMP TEST: ")
    print(test_err_str)
 
    
 # =============================================================================
 # Function to implement one iteration of adaboost on the train data.
 # Trains a decision stump, then calculates the predictions, error and vote, then finds the final hypothesis.
 # ===========================================================================
def _train_iter_ada_boost(t, y):

    global vote_arr
    s = ID3.train_data
    tree = ID3.train(s, t)
    trees.append(tree)

    predictions[t] = _get_predictions(s, tree.root, predictions[t])
    error = _get_error(y, t)

    vote_arr.append(_get_vote(error))
    votes = np.array(vote_arr)
    if t != T-1:  _get_weights(votes, y, t)

    hyp = _get_adaboost_hyp(votes, predictions)
    return [hyp, votes]
# =============================================================================
# Function to implement the final hypothesis of the test data when run on the trained trees
# ===========================================================================

def _test_iter_ada_boost(votes, _predictions):
    
    s = ID3.test_data
    for t in range(len(trees)):
        _predictions[t] = _get_predictions(s, trees[t].root, _predictions[t])
    return _get_adaboost_hyp(votes, _predictions)

# ===========================================================================
# Function to Calculates the error for predictions[t] with example_wt[t]
# ===========================================================================
def _get_error(y, t):
    
    return 0.5 - (0.5 * (np.sum(ID3.example_wt[t] * y * predictions[t])))

# ===========================================================================
# Function to Calculates the vote for the given error
# ===========================================================================
def _get_vote(error):

    
    return 0.5 * math.log((1.0 - error) / error)

# ===========================================================================
# Function to Calculates the weights for the adaboost algorithm
# ===========================================================================
def _get_weights(votes, y, t):
    
    ID3.example_wt[t+1] = ID3.example_wt[t] * np.exp(-votes[t] * y * predictions[t])
    z = np.sum(ID3.example_wt[t+1])
    ID3.example_wt[t+1] /= z

# ===========================================================================
# Function to Sums up the predictions times the given votes
# ===========================================================================
def _get_adaboost_hyp(votes, _predictions):
  
    temp = np.tile(np.zeros(m), (len(votes), 1))
    for index in range(len(votes)):
        temp[index] = np.array(votes[index] * _predictions[index])
    return np.sign(temp.sum(axis=0))

# =============================================================================
# #Function to implement bagged decision tree algorithm for T different samples

# ============================================================================= 
def _run_bagged():

    global predictions
    predictions = np.zeros(4999)
    test_predictions = np.zeros(4999)
    train_err_str = ""
    test_err_str = ""

    for t in range(T):
        [train_err, predictions] = _run_bagged_iter(t, True, ID3.train_data, predictions)
        train_err_str += str(train_err) + ","

        [test_err, test_predictions] = _run_bagged_iter(t, False, ID3.test_data, test_predictions)
        test_err_str += str(test_err) + ","

    print("TRAIN: ")
    print(train_err_str)
    print("TEST: ")
    print(test_err_str)

# =============================================================================
# #Function to implement one iteration of the bagged decision tree algorithm

# =============================================================================

def _run_bagged_iter(_t, is_train, data, _predictions):
    
    s = _random_sample(data)
    if is_train: trees.append(ID3.train(s, _t, attr_subset_num))
    _predictions = _get_bagged_pred(data, trees[_t].root, _predictions)
    hyp = _get_bagged_final_hyp(_predictions)

    return [_get_PE(np.array(data[-1], dtype=int), hyp), _predictions]
# =============================================================================
# #Function to Draws m samples uniformly with replacement

# =============================================================================

def _random_sample(data):
   
    s = []
    indices = []
    for i in range(len(data)):
        s.append([])
    for i in range(m):
        n = random.randint(0, len(data[-1]) - 1)
        indices.append(n)
        for j in range(len(data)):
            s[j].append(data[j][n])
    return s
# =============================================================================
# #Function to Updates predictions that are 0 to -1 or 1

# =============================================================================
def _get_bagged_final_hyp( _predictions):

    is_even = True
    final_hyp = np.sign(_predictions)
    for i in range(_predictions.size):
        p = _predictions[i]

        if p == 0:
            if is_even: p = 1
            else: p = -1
            is_even = not is_even

        final_hyp[i] = p
    return np.sign(final_hyp)



# =============================================================================
# #Function to implement bias and variance
#     # TODO: BIAS
    # bias_terms = 0.0
    # for row in range(len(_trees)):
    # average( h_i(x*)) - f(x*) ) ^2.
# =============================================================================
def _bias_variance_decomp():
    m = 1000
    _trees = []
    for i in range(100):
        trees.append([])

        for t in range(T):
            s = _random_sample(ID3.train_data)
            trees[i].append(ID3.train(s, t))
            if t == 0: _trees.append(trees[i][t])

    test_predictions = np.zeros((100, m))
    y = np.array(ID3.test_data[-1], dtype=int)
    for i in range(len(_trees)):
        test_predictions[i] = _get_predictions(ID3.test_data, _trees[i].root, test_predictions[i])



# =============================================================================
# #Function to implement random forest algorithm
# Runs T iterations of random forest for each attribute subset size in [2, 4, 6]
# =============================================================================
def _run_random_forest():
    
    global attr_subset_num, trees
    for attr_subset_num in attr_subset_nums:
        trees = []
        print("attribute subset size: ", attr_subset_num, ": ")
        _run_bagged()


# =============================================================================
# #Function to Calculates the predictions for the given tree root 
# and using predict_example()
# =============================================================================
def _get_predictions(s, root, _predictions):

    p = _predictions.copy()
    for index in range(len(s[-1])):
        example = []
        for l in s:
            example.append(l[index])
        p[index] = ID3.predict_example(example, root, False)
    return p

# =============================================================================
# #Function to Use the initial indexes of the sample data
# to calculate the overall prediction of all examples in s
# =============================================================================
def _get_bagged_pred(data, root, _predictions):
    """"""
    round_predictions = np.zeros(len(data[-1]))
    round_predictions = _get_predictions(data, root, round_predictions)
    for i in range(len(data[-1])):
        _predictions[i] += round_predictions[i]
    return _predictions

# =============================================================================
# #Function to Calculates the percentage of incorrect predictions
# =============================================================================
def _get_PE(y, _predictions):

    count = 0
    for i in range(len(y)):
        if y[i] != _predictions[i]: count += 1
    return count / len(y)


if __name__ == '__main__':
    ID3.DL_select = "bank"
    alg_type = sys.argv[1]

    if alg_type == "ada":
        ID3.max_depth = 2
        predictions = np.zeros((T, m))
    else:
        predictions = np.zeros(4999)
        m = 2500

    ID3.setup_data(m, T)

    if alg_type == "ada":
        print("ADA BOOST")
        _run_ada_boost()

    elif alg_type == "bag":
        print("BAGGED DECISION TREES")
        _run_bagged()

    elif alg_type == "forest":
        print("RANDOM FOREST")
        _run_random_forest()


