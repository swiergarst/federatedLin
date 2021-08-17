from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
from vantage6.client import Client










def master(client, data, id_array, input_array):
    own_id = client.organization["id"]
    index = np.where(id_array == own_id)
    return input_array[index] * 2

    

def RPC_train_and_test(data, model, classes, use_scaffold, c, ci):

    dim_num = 784
    dims = ['pixel' + str(i) for i in range(dim_num)]
    X_train_arr = data.loc[data['test/train'] == 'train'][dims].values
    y_train_arr = data.loc[data['test/train'] == 'train']['label'].values
    X_test_arr = data.loc[data['test/train'] == 'test'][dims].values
    y_test_arr = data.loc[data['test/train'] == 'test']['label'].values

    old_coef = model.coef_
    old_inter = model.intercept_


    model.partial_fit(X_train_arr, y_train_arr, classes=classes)
    new_coef = model.coef_
    new_inter = model.intercept_

    # bit of a hack to 'test before train'
    model.coef_  = old_coef
    model.intercept_ = old_inter
    result = model.score(X_test_arr, y_test_arr)
    
 
    return(result, new_coef, new_inter)