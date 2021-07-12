from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np











def master():
    pass

def RPC_train_and_test(data, model):

    dim_num = 784
    dims = ['pixel' + str(i) for i in range(dim_num)]
    X_train_arr = data.loc[data['test/train'] == 'train'][dims].values
    y_train_arr = data.loc[data['test/train'] == 'train']['label'].values
    X_test_arr = data.loc[data['test/train'] == 'test'][dims].values
    y_test_arr = data.loc[data['test/train'] == 'test']['label'].values

    model.partial_fit(X_train_arr, y_train_arr, classes=[0,1])

    result = model.score(X_test_arr, y_test_arr)

    return(result, model.coef_, model.intercept_)