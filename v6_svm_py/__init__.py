from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
from vantage6.client import Client










def master(client, data, id_array, input_array):
    own_id = client.host_node_id
    index = np.where(id_array == own_id)
    id = client.request(f'node/{own_id}/organization')
    return id

    

def RPC_train_and_test(data, model, classes, use_scaffold, c, ci):

    dim_num = 784
    dims = ['pixel' + str(i) for i in range(dim_num)]
    X_train_arr = data.loc[data['test/train'] == 'train'][dims].values
    y_train_arr = data.loc[data['test/train'] == 'train']['label'].values
    X_test_arr = data.loc[data['test/train'] == 'test'][dims].values
    y_test_arr = data.loc[data['test/train'] == 'test']['label'].values

    old_coef = np.copy(model.coef_)
    old_inter = np.copy(model.intercept_)


    model.partial_fit(X_train_arr, y_train_arr, classes=classes)

    if use_scaffold:
        model.coef_ += + c["coef"] - ci["coef"]
        model.intercept_ += + c["inter"] - ci["inter"]
        ci["coef"] = ci["coef"] - c["coef"] + (1/model.get_params()['eta0']) * (old_coef - model.coef_)
        ci["inter"] = ci["inter"] - c["inter"] + (1/model.get_params()['eta0']) * (old_inter - model.intercept_)

    new_coef = np.copy(model.coef_)
    new_inter = np.copy(model.intercept_)

    # bit of a hack to 'test before train'
    model.coef_  = np.copy(old_coef)
    model.intercept_ = np.copy(old_inter)
    result = model.score(X_test_arr, y_test_arr)
    
 
    return(result, new_coef, new_inter, ci, X_train_arr.shape[0])