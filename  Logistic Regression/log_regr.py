import numpy as np
import os,copy
import pandas as pd
from sklearn.preprocessing import normalize
import func_log_regr



var = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
# total epoch
T = 10

def Load_dataset(path):
    X = pd.read_csv(path, header=None)
    n_example = X.shape[0]
    n_feat = X.shape[1]
    X[n_feat-1] = X[n_feat-1].apply(lambda val: -1 if val==0 else 1)
    data = []
    y = []

    for i in range(n_example):
        dp = X.iloc[i].tolist()
        
        x = copy.deepcopy(dp[0:n_feat])
        x[-1] = 1
        data.append(x)

        y.append(dp[n_feat-1])

    return np.array(data), np.array(y)

if __name__ == "__main__":


    Xtr, ytr = Load_dataset('./bank-note/train.csv')
    Xte, yte = Load_dataset('./bank-note/test.csv')
    total_variation = len(var) * 2
    print(f"Setting with {total_variation} variations...")

    for v in var:
        print(f"Var={v}")

        ML = func_log_regr.Log_reg_algo(len(Xtr[0]))
        ML.train(Xtr, ytr, d=0.1, lr=0.01, T=T)

        preds = ML.get_pred(Xtr)
        err_tr = 0
        for i,p in enumerate(preds):
            if p != ytr[i]:
                err_tr += 1
        

        preds = ML.get_pred(Xte)
        err_te = 0
        for i,p in enumerate(preds):
            if p != yte[i]:
                err_te += 1
        print("MLE Results")
        print(f"\tMLE Training Error: {err_tr / Xtr.shape[0]} ")
        print(f"\tMLE Testing Error: {err_te / Xte.shape[0]} ")

        MAP = func_log_regr.Log_reg_algo(len(Xtr[0]), "map")
        MAP.train(Xtr, ytr, d=0.1, lr=0.01, var=v, T=T)

        preds = MAP.get_pred(Xtr)
        err_tr = 0
        for i,p in enumerate(preds):
            if p != ytr[i]:
                err_tr += 1

        preds = MAP.get_pred(Xte)
        err_te = 0
        for i,p in enumerate(preds):
            if p != yte[i]:
                err_te += 1
        print("MAP Results")
        print(f"\tMAP Training Error: {err_tr / Xtr.shape[0]} ")
        print(f"\tMAP Testing Error: {err_te / Xte.shape[0] }")

    
    
