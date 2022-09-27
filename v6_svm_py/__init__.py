from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
import math
from vantage6.client import Client










def master(client, data, id_array, input_array):
    own_id = client.host_node_id
    index = np.where(id_array == own_id)
    id = client.request(f'node/{own_id}/organization')
    return id

    

def RPC_train_and_test(data, model, classes, use_scaffold, c, ci, num_local_rounds, num_local_batches, weighted_lr = False, lr_pref = None):


    X_train_arr = data.loc[data['test/train'] == 'train'].drop(columns = ['test/train', 'label']).values
    y_train_arr = data.loc[data['test/train'] == 'train']['label'].values
    X_test_arr = data.loc[data['test/train'] == 'test'].drop(columns = ["test/train", "label"]).values
    y_test_arr = data.loc[data['test/train'] == 'test']['label'].values

    if weighted_lr:
        dset_size = X_train_arr.shape[0]
        lr_local = lr_pref * dset_size
        model.set_params({"eta0" : lr_local})

 
    batch_size = math.floor(X_train_arr.shape[0]/num_local_batches)
    

    #if use_dgd:
    #    model.coef_ = np.mean(nb_parameters["coef"], axis =0)
    #    model.intercept_ = np.mean(nb_parameters["inter"], axis = 0)

    old_coef = np.copy(model.coef_)
    old_inter = np.copy(model.intercept_)


    for round in range(num_local_rounds):
        for batch in range(num_local_batches):
            X_train_b = X_train_arr[batch*batch_size:(batch+1) *batch_size]
            y_train_b = y_train_arr[batch*batch_size:(batch+1) *batch_size]
            model.partial_fit(X_train_b, y_train_b, classes=classes)




        if use_scaffold:
            m_copy = np.copy(model.coef_)
            m_copy2 = np.copy(model.intercept_)
            lr = model.get_params()['eta0']

            model.coef_ = m_copy - lr * (c["coef"] - ci["coef"]) 
            model.intercept_ = m_copy2 - lr * (c["inter"] - ci["inter"])
            ci["coef"] = ci["coef"] - c["coef"] + (1/(lr* num_local_batches)) * (old_coef - m_copy)
            ci["inter"] = ci["inter"] - c["inter"] + (1/(lr* num_local_batches)) * (old_inter - m_copy2)


    new_coef = np.copy(model.coef_)
    new_inter = np.copy(model.intercept_)

    # bit of a hack to 'test before train'
    model.coef_  = np.copy(old_coef)
    model.intercept_ = np.copy(old_inter)
    result = model.score(X_test_arr, y_test_arr)
        
 
    return(result, new_coef, new_inter, ci, dset_size)