from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
from vantage6.client import Client










def master(client, data, id_array, input_array):
    own_id = client.host_node_id
    index = np.where(id_array == own_id)
    id = client.request(f'node/{own_id}/organization')
    return id

    

def RPC_train_and_test(data, model, classes, use_scaffold, c, ci, num_local_rounds):


    X_train_arr = data.loc[data['test/train'] == 'train'].drop(columns = ['test/train', 'label']).values
    y_train_arr = data.loc[data['test/train'] == 'train']['label'].values
    X_test_arr = data.loc[data['test/train'] == 'test'].drop(columns = ["test/train", "label"]).values
    y_test_arr = data.loc[data['test/train'] == 'test']['label'].values

    old_coef = np.copy(model.coef_)
    old_inter = np.copy(model.intercept_)

    for round in range(num_local_rounds):

        model.partial_fit(X_train_arr, y_train_arr, classes=classes)

        if use_scaffold:

            m_copy = np.copy(model.coef_)
            m_copy2 = np.copy(model.intercept_)
            lr = model.get_params()['eta0']

            model.coef_ = m_copy - lr * (c["coef"] - ci["coef"]) 
            model.intercept_ = m_copy2 - lr * (c["inter"] - ci["inter"])
            #print(model.coef_.shape)
            #model.coef_ += c["coef"] - ci["coef"]
            ci["coef"] = ci["coef"] - c["coef"] + (1/lr) * (old_coef - m_copy)
            ci["inter"] = ci["inter"] - c["inter"] + (1/lr) * (old_inter - m_copy2)
            #ci["coef"] = np.copy(ci["coef"] - c["coef"]) + (1/model.get_params()['eta0']) * np.copy((old_coef - model.coef_))
            #ci["inter"] = np.copy(ci["inter"] - c["inter"]) + (1/model.get_params()['eta0']) * np.copy((old_inter - model.intercept_))

        new_coef = np.copy(model.coef_)
        new_inter = np.copy(model.intercept_)

    # bit of a hack to 'test before train'
    model.coef_  = np.copy(old_coef)
    model.intercept_ = np.copy(old_inter)
    result = model.score(X_test_arr, y_test_arr)
        
 
    return(result, new_coef, new_inter, ci, X_train_arr.shape[0])