
# Decision Tree implementation using ID3 Algorithm
# Step 1: Load Dataset
# Step 2: Prediction error: Entropy, Gini Index and Majority Error
# Step 3: feature Index: Information Gain
# The max tree max_depth is also varied. 


import sys
import math
import Data # module to get data and pre-processed data
import numpy as np
import random


# Variables
DL_select = "bank"
feature = "ig"
max_depth = -1
#train_data = Data.load_data(DL_select, True, True)

# # Decision Tree class: Define tree structure
# _init_ function:
# input:
#     :param: max_depth : path to import sample data (s)
#           : feature : type of dataset
# _set_root function:
#             :param: node list



class Tree:
    def __init__(self, feature="ig"):
        self.feature = feature
        self.max_depth = max_depth
        self.root = None

    def set_root(self, node):
        self.root = node


# # Decision Tree class: Define tree structure
# _init_ function:
# input:
#     :param: max_depth : path to import sample data (s)
#           : feature : type of dataset
# _set_root function:
#             :param: node list

               
class Node:
    def __init__(self, s,_example_wt, parent, is_leaf):
        self.s = s
        self.example_wt = _example_wt
        self.parent = parent
        self.is_leaf = is_leaf
        self.branches = {}
        self.attribute = -1
        self.label = None

    def set_attribute(self, attribute):
        self.attribute = attribute

    def set_label(self, label):
        self.label = label

    def add_branch(self, value, node):
        self.branches[value] = node


# #Helper Function to help __main_

def _run():
    """ Runs the normal ID3 algo"""
    tree = train(train_data, 0)
    print("TRAIN:", predict(train_data, tree.root))
    print("TEST: ", predict(test_data, tree.root))

# Gets a subset of attributes for the next attribute split
# if there is a set subset size.
# Otherwise just returns the unused attributes
# input:
#     :param: attribute : attributes
# output:
#     :param: attribute : attributes

def _get_attr(_attributes):
    if attr_subset_num != 0:
        return _get_subset_of_attr(_attributes)
    else:
        return _attributes

# Gets a random subset of attributes that is not splitted before 
# input:
#     :param: attribute : attributes
# output:
#     :param: subset : list

def _get_subset_of_attr(_attributes):
    subset = []
    while len(subset) < attr_subset_num and len(subset) < len(_attributes):
        n = random.randint(0, len(attributes) - 1)
        if attributes[n] in _attributes:
            subset.append(attributes[n])
    return subset

# Train a descision tree using ID3 based on given feature 
# input:
#     :param s: [](examples, features) - the entire dataset
#     :param t: the index of example_weights to use. ie the training iteration.
#     :param _attr_subset_num: If running Random Forest, set the attribute subset size.
# output:
#     :param: tree :The trained tree

def train(s, t, _attr_subset_num=0):
    global attr_subset_num
    attr_subset_num = _attr_subset_num

    _tree = Tree(feature)
    _tree.set_root(_recursive_ID3(s, example_wt[t].copy(), None, attributes.copy(), 1))
    return _tree

# #ID3 Ref: Gain(S,A) = Feature(S)- ratio* Feature(S_V)
# Feature: Entropy/ME/GI
# ratio : Sum_v<A((S_v/S))


def _recursive_ID3(s, _example_wt, parent, _attributes,level):
    if s[-1].count(s[-1][0]) == len(s[-1]):
        node = Node(s, _example_wt, parent, True)
        node.set_label(s[-1][0])
        return node

    elif len(_attributes) == 0 or level == max_depth:
        node = Node(s, _example_wt, parent, True)
        node.set_label(_select_MajLabel(s[-1], _example_wt))
        return node

    else:
        node = Node(s, _example_wt, parent, False)
        _split_on_gain(node, _get_attr(_attributes))

        for value in node.attribute:
            arr = _find_s_v(node, node.attribute, value)
            s_v = arr[0]
            _example_wt_v = np.array(arr[1])

            if len(s_v[-1]) == 0:
                label = _select_MajLabel(s[-1], _example_wt)
                child = Node({}, np.array([]), node, True)
                child.set_label(label)
                node.add_branch(value, child)

            else:
                a = _attributes.copy()
                a.remove(node.attribute)
                child = _recursive_ID3(s_v, _example_wt_v, node, a, level + 1)
                node.add_branch(value, child)

        return node

# #Helper Function to compute s_v sample to specific value
# input:
#     :param: node: node list
#     :param: attribute: attribute
#     :param: value: attribute value
# output:   :return: Calculated s_v, example list


def _find_s_v(node, attribute, value):
    attr_idx = attributes.index(attribute)
    indices = [i for i, x in enumerate(node.s[attr_idx]) if x == value]
    s_v = node.s.copy()

    for i in range(len(node.s)):
        new_feature_list = []

        for index in indices:
            new_feature_list.append(node.s[i][index])
        s_v[i] = new_feature_list

    example_list = []
    for index in indices:
        example_list.append(node.example_wt[index])

    return [s_v, np.array(example_list)]


# #Helper Function to compute feature value
# input:
#     :param: s: sample list
#     :param: label: target attribute s[-1]
# output:   :return: Calculate lenth of sample on specific label

def _split_on_gain(node, _attributes):
    G = []
    for i in _attributes:
        G.append(_calculate_gain(node,i))
    max_index = G.index(max(G))
    node.set_attribute(_attributes[max_index])

# # Function: This function computes gain for given sample and attribute
# input:
#     :param: node: split on that node 
#           : attribute : attribute at that split one
#           
# output:   :return: Gain(S,A) for particular feature
     
def _calculate_gain(node, attribute):
    gain = 0.0
    gain += _select_feature(node.s, node.example_wt)
    for value in attribute:
        arr = _find_s_v(node, attribute, value)
        s_v = arr[0]
        _example_wt_v = np.array(arr[1])

        if len(s_v[-1]) != 0:
            ratio = np.sum(_example_wt_v)
            feat_s_v = _select_feature(s_v, _example_wt_v)

            if feat_s_v != 0:
                gain -= ratio * feat_s_v

    return gain

# #Helper Function to compute gain on specific feature
# input:
#     :param: s: sample list
#     : param : example: given set of example 
# output:   :return: Calculate feature value on selection for input feature
 

def _select_feature(s, _example_wt):
    if   feature == "me": return _calculate_ME(s, _example_wt)
    elif feature == "gi": return _calculate_GI(s, _example_wt)
    else: return _calculate_H_s(s, _example_wt)

# #Helper Function to compute gain on specific feature
# input:
#     :param: s: sample list
#     : param : example: given set of example 
# output:   :return: Calculate Entropy 


def _calculate_H_s(s, _example_wt):
    h_s = 0.0
    for label in labels:
        probability_of_label = _get_s_l(s, label, _example_wt)
        if probability_of_label != 0:
           
            h_s -= probability_of_label * math.log(probability_of_label, 2)
    return h_s

# #Helper Function to compute gain on specific feature
# input:
#     :param: s: sample list
#     : param : example_wt: given set of example 
# output:   :return: Calculate Majority Error if "me" selected


def _calculate_ME(s, _example_wt):
    Maj_l = _select_MajLabel(s[-1], _example_wt)
    me = 1 - _get_s_l(s, Maj_l, _example_wt)
    return me


# #Helper Function to compute feature value
# input:
#     :param: s_l: sample list 
#     
# output:   :return: majority label

def _select_MajLabel(y, _example_wt):
    count = [0 for _ in range(len(labels))]
    for i in range(len(y)):
        label = y[i]
        for j in range(len(labels)):
            if label == labels[j]:
                count[j] += _example_wt[i]
                break

    index = count.index(max(count))
    return labels[index]

# #Helper Function to compute gain on specific feature
# input:
#     :param: s: sample list
#     : param : example_wt: given set of example 
# output:   :return: Calculate Gini Index if "gi" selected

def _calculate_GI(s, _example_wt):
    gi = 1.0
    for label in labels:
        num_s_l = _get_s_l(s, label, _example_wt)
        if num_s_l != 0:
            p_l = num_s_l 
            gi -= p_l**2
    return gi


# #Helper Function to compute feature value
# input:
#     :param: s: sample list
#     :param: label: target attribute s[-1]
# output:   :return: Calculate lenth of sample on specific label

def _get_s_l(s, label, _example_wt):
    total = 0.0
    for i in range(len(s[-1])):
        if s[-1][i] == label:
            total += _example_wt[i]
    return total / np.sum(_example_wt)


# #Function to train a decision tree with the given data, the ID3 algorithm, and the type of feature function given
# input:
#     :param: s: data list
#     :param: root: root node
# output:   :return pred error 

def predict(s, root):

    corr = 0
    for index in range(len(s[-1])):
        example = []
        for l in s:
            example.append(l[index])
        corr += predict_example(example, root,True)

    return corr/len(s[-1])


# #Function to predicts the given sample recursively
# input:
#     :param: example : [features] - The feature values for a single example
#     :param node: root node.
#     :param is_id3: True :Id3 False:bag/boost
# output:   :return 0 - correct. 1 - incorrect


def predict_example(example, node, is_id3):

    if not node.is_leaf:
        attr_idx = attributes.index(node.attribute)
        child = node.branches[example[attr_idx]]
        return predict_example(example, child, is_id3)
    else:
        if is_id3:
            if node.label == example[-1]: return 0
            else: return 1
        else:
            return node.label

# #Function to implement recursive check tree
# input:
#     :param: node: root node list
#     :param: attr: attribute list
#     :param: branches: branch list
#     :param: level: max_depth
# output:   :check tree and prints out attributes, branches, it took to get to a label


def _check_tree(node, _attributes=[], branches=[], level=0):

    if node.is_leaf:
        _pr_attr = ""
        _pr_brnch = ""
        for _a in _attributes:
            _pr_attr += str(_a) + ", "
        for b in branches:
            _pr_brnch += b + ", "
        print("ATTRIBUTES: ", _pr_attr, "BRANCHES: ", _pr_brnch, "LABEL: ", node.label, "LEVEL: ", level)

    else:
        _attributes.append(node.attribute)
        for branch, child in node.branches.items():
            copy = branches.copy()
            copy.append(branch)
            _check_tree(child, _attributes.copy(), copy, level+1)



# Function to set up attributes, labels, example_weights train_data, and
# test_data based on the data_type 
# input:
#     :param: m : number of examples ID3 will run on
#     :param iters: number of iterations ID3 will be run

def setup_data(m=4999, iters=1):
    global attributes, labels, example_wt, train_data, test_data
    if DL_select == "car":
        attributes = Data.car_attributes
        labels = Data.car_labels
        m = 1000

    else:
        attributes = Data.bank_attributes
        labels = Data.bank_labels
    example_wt = np.tile(np.repeat(1.0 / m, m), (iters, 1))
    train_data = Data.load_data(DL_select, True, True)
    test_data = Data.load_data(DL_select, False, True)

# #Main function
# 
# output:   :call several helper finctio to build tree using ID3
  
if __name__ == '__main__':
    
    DL_select = sys.argv[1]
    feature = sys.argv[2]
    if len(sys.argv) > 3:
        max_depth = int(sys.argv[3])

    setup_data()
    _run()

    

    


    





    

    

