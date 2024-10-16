
#Author: Ekata Mitra
# =============================================================================
# CS 5350/6350: Machine Learining Fall 2022:
# HW2: Section 2
# Question 2 and 3
# =============================================================================

# =============================================================================
# Step 1: Visualize Dataset (car, bank)
# Step 2: Several Helper Function 

# =============================================================================

import statistics
import numpy as np
# =============================================================================
# Step 1: Visualize Dataset (car, bank)

# Car dataset
# =============================================================================
# list of attributes
car_attributes = [
    ["vhigh", "high", "med", "low"],
    ["vhigh", "high", "med", "low", "."],
    ["2", "3", "4", "5more"],
    ["2", "4", "more"],
    ["small", "med", "big"],
    ["low", "med", "high"]
]

car_labels = ["unacc", "acc", "good", "vgood"]

bank_attributes = [
    ["numeric", "leq", "over"],
    ["job", "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
        "blue-collar", "self-employed", "retired", "technician", "services"],
    ["marital", "married", "divorced", "single"],
    ["education", "unknown", "secondary", "primary", "tertiary"],
    ["default", "yes", "no"],
    ["numeric", "leq", "over"],
    ["housing", "yes", "no"],
    ["loan", "yes", "no"],
    ["contact", "unknown", "telephone", "cellular"],
    ["numeric", "leq", "over"],
    ["month", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    ["numeric", "leq", "over"],
    ["numeric", "leq", "over"],
    ["numeric", "leq", "over"],
    ["numeric", "leq", "over"],
    ["poutcome", "unknown", "other", "failure", "success"]
]

bank_labels = [-1, 1]

example_attributes = [["s", "o", "r"], ["h", "m", "c"], ["h", "n", "l"], ["s", "w"]]

# =============================================================================
# # Step 2: Several Helper Functions
# =============================================================================

# =============================================================================
# # Function: This function imports the data from a csv file as list
# input:
#     :param: dir_loc: dir_loc to import sample data (s)
#           : train : type of dataset
# output:   :return: the data as a list of lists that contain all the example values
#                    for an attribute or label (at s[-1]).
# =============================================================================
def load_data(dir_loc, train,param):
    s =[] 
    if train:
        CSVfile = "/Users/pratishthaagnihotri/Documents/Machine_Learning/HW_2/" + dir_loc + "/train.csv"
    else:
        CSVfile = "/Users/pratishthaagnihotri/Documents/Machine_Learning/HW_2/" + dir_loc + "/test.csv"
        
    with open(CSVfile, 'r') as f:
        
        num_tot_attribute = 0 
        for line in f:
            terms = line.strip().split(',')
            num_tot_attribute = len(terms)
            break
        s = [[] for _ in range(num_tot_attribute)]

        for line in f:
            terms = line.strip().split(',')
            for i in range(num_tot_attribute):

                if dir_loc== "bank" and i == num_tot_attribute - 1:
                    if terms[i] == "yes": s[i].append(1)
                    else: s[i].append(-1)

                else: s[i].append(terms[i])
    if dir_loc == "bank":
        attributes = bank_attributes
        temp = _change_NumAttr_to_BinAttr(s, attributes)
        if not param:
            s = _change_MissAttr_to_MajAttr(s, attributes, train)
    return s

# =============================================================================
# # Function: - Check all numeric attributes
#             - Compute the median
#             - Update the attributes to contain the median
#             - Update all example to "leq", for equal to or less 
#             - Update "over" for the numeric attributes.
# input:
#     :param: s: the entire dataset
#     :param attributes: all attributes     
# output:    
#           return: the updated dataset
# =============================================================================

def _change_NumAttr_to_BinAttr(s, attributes):
    for i in range(len(attributes)):
        if attributes[i][0] == "numeric":
            median = _get_median(s[i])
            attributes[i][0] = str(median)
            s[i] = _update_NumAttr(s[i], attributes[i])

        elif _is_NumAttr(attributes[i]):
            s[i] = _update_NumAttr(s[i], attributes[i])
    return s

# =============================================================================
# # Helper Function to  _change_NumAttr_to_BinAttr for Bank dataset:
#             - Check bank_attributes is numeric or not
#
# input:
#     :param: attribute: the attrbute list
#         
# output:    
#           return: boolean "True" or "False"
# =============================================================================
def _is_NumAttr(attribute):
    try:
        int(attribute[0])
        return True
    except ValueError:
        return False

# =============================================================================
# # Helper Function to  _change_NumAttr_to_BinAttr for Bank dataset:
#             - Compute the median of the set
#
# input:
#     :param: val_a : value at a numeric attribute
#         
# output:    
#           return: median 
# =============================================================================

def _get_median(val_a):
    int_val = list(map(int, val_a))  # str to num
    median = statistics.median(int_val)
    return median

# =============================================================================
# # Helper Function to  _change_NumAttr_to_BinAttr for Bank dataset:
#             - Update all example to "leq", for equal to or less 
#             - Update "over" for the numeric attributes.
# input:
#     :param: val_a : value at a numeric attribute
#     :param: attribute: the attrbute list    
# output:    
#           return: binary attribute 
# =============================================================================

def _update_NumAttr(val_a, attribute):
    for i in range(len(val_a)):
        if int(val_a[i]) > int(attribute[0]): val_a[i] = "over"
        else: val_a[i] = "leq"
    return val_a

# =============================================================================
# # Function: - Check all missing attributes["unknown"]
#             - Compute the majority
#             - Update the missing attributes to majority attribute
# input:
#     :param: s: the entire dataset
#     :param: attributes: all attributes
#     :param: DL_select_type: train only
# output:    
#           return: the updated dataset
# =============================================================================

def _change_MissAttr_to_MajAttr(s, attributes, train):
    major = []
    for i in range(len(attributes)):

        if train:
            major.append("")
            if "unknown" in attributes[i]:
                majority_attribute = _find_MajAttr(s[i], attributes[i])
                major[i] = majority_attribute

                for j in range(len(s[i])):
                    if s[i][j] == "unknown":
                        s[i][j] = majority_attribute

        elif "unknown" in attributes[i]:
            for j in range(len(s[i])):
                if s[i][j] == "unknown":
                    s[i][j] = major[i]

    return s

# =============================================================================
# # Helper Function: 
#             - Compute the majority
# input:
#     :param: val_a : value at a numeric attribute
#     :param: attribute: the attrbute list 
# output:    
#           return: majority attribute value
# =============================================================================

def _find_MajAttr(val_a, attribute):
    
    cnt = [0 for _ in range(len(attribute))]

    for val in val_a:
        for i in range(len(attribute)):

            if val == attribute[i] and attribute[i] != "unknown":
                cnt[i] += 1
                break

    inx = cnt.index(max(cnt))
    return attribute[inx]

# visualize car example 
def small_car_data():

    return [
        ["low", "med", "high"],
        ["high", "high", "high"],
        ["5more", "5more", "5more"],
        ["4", "4", "4"],
        ["med", "med", "med"],
        ["high", "high", "high"],
        ["vgood", "good", "acc"]
    ] # 3 examples

# visualize bank example

def small_bank_data():
    s = [
        ["48","48","53"],
        ["services","blue-collar","technician"],
        ["married", "married", "married"],
        ["secondary", "secondary", "secondary"],
        ["no", "no", "no"],
        ["0", "0", "0"],
        ["yes", "yes", "yes"],
        ["no", "no", "no"],
        ["unknown", "unknown", "unknown"],
        ["5", "5", "5"],
        ["may", "may", "may"],
        ["114", "114", "114"],
        ["2", "2", "2"],
        ["-1", "-1", "-1"],
        ["0", "0", "0"],
        ["unknown", "unknown", "unknown"],
        ["no", "no", "yes"],
    ]
    
    for i in range(len(s[-1])):
        if s[-1][i] == "yes":
            s[-1][i] = 1
        else:
            s[-1][i] = 0
    s = _change_NumAttr_to_BinAttr(s, bank_attributes)
    return s

def get_example_data():
# 4 examples    
    s = [
        ["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"],
        ["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"],
        ["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"],
        ["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"],
        ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"],
    ]
    
    for i in range(len(s[-1])):
        if s[-1][i] == "yes":
            s[-1][i] = 1
        else:
            s[-1][i] = -1

    return s
