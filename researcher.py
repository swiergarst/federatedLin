# with lots of inspiration from the courselabs from the CS4240 deep learning course and fedML

### imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import time 
import pandas as pd
from vantage6.tools.util import info
from v6_svm_py.lin_config_functions import init_model

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
#sys.path.insert(1, os.path.join(sys.path[0], '../../..'))

from sklearn.linear_model import SGDClassifier
from io import BytesIO
from vantage6.client import Client
from helper_functions import heatmap
from comp_functions import average, scaffold
from config_functions import get_full_dataset, get_datasets, clear_database, get_save_str
start_time = time.time()
### connect to server


print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = "/home/swier/.local/share/vantage6/node/privkey_testOrg0.pem"
client.setup_encryption(privkey)




#organizations = client.get_organizations_in_my_collaboration()
#org_ids = [organization["id"] for organization in organizations]



ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

### parameter settings ###

#learning rates
lr_local = 5e-5
lr_global = 1 #only affects scaffold. 1 is recommended


#dataset and booleans
dataset = 'fashion_MNIST' #options: MNIST_2class, MNIST_4class, fashion_MNIST, A2_PCA, 3node
week = "afstuderen/datafiles/svm/dgd/" #folder for saving data files 
classifier = "SVM" #either SVM or LR

save_file = True # whether to save results in .npy files
use_scaffold = False # if true, the SCAFFOLD algorithm is used (instead of federated averaging)
use_dgd = True


# these settings change the distribution of the datasets between clients. sample_imbalance is not checked if class_imbalance is set to true
class_imbalance = True
sample_imbalance = False 

use_sizes = True # if set to false, the unweighted average is taken instead of the weighted average

save_str = get_save_str(dataset, classifier, class_imbalance, sample_imbalance, use_scaffold, use_sizes, lr_local, 1, 1, use_dgd)

#federated settings
num_global_rounds = 100 #communication rounds
num_local_epochs = 1 #local epochs between each communication round (currently not implemented)
num_local_batches = 1
num_clients = 10 #amount of federated clients (make sure this matches the amount of running vantage6 clients)
num_runs =  4  #amount of experiments to run using consecutive seeds
seed_offset = 0 #decides which seeds to use: seed = seed_offset + current_run_number





'''
A_alt = np.array([[0,1,9],
                [1,0,2],
                [2,1,3],
                [3,2,4],
                [4,3,5],
                [5,4,6],
                [6,5,7],
                [7,6,8],
                [8,7,9],
                [9,8,0]])
'''
### end of settings ###

if classifier == "SVM":
    loss = "hinge"
elif classifier == "LR":
    loss = "log"
else:
    raise(Exception("unknown classifier provided"))


# data structures to store results
c_log = np.zeros((num_global_rounds))
ci_log = np.zeros((num_global_rounds, num_clients))

prevmap = heatmap(num_clients, num_global_rounds)
newmap = heatmap(num_clients, num_global_rounds)
cmap = heatmap(num_clients , num_global_rounds)
datasets = get_datasets(dataset)
X_test, y_test = get_full_dataset(datasets, "FNN")
X_test = X_test.numpy()
y_test = y_test.numpy()

### main loop
for run in range(num_runs):

    # generation of ring graph (neighbours generated randomly)
    clients_seq = np.arange(num_clients)

    np.random.shuffle(clients_seq)

    A_alt = np.zeros((num_clients, 3), dtype=int)

    for client_i, node in enumerate (clients_seq):
        A_alt[node,0] = node
        A_alt[node,1] = clients_seq[client_i - 1]
        A_alt[node,2] = clients_seq[(client_i + 1) % num_clients]

    accuracies = np.zeros(( num_clients, num_global_rounds))
    global_accuracies = np.zeros((num_global_rounds))
    seed = run + seed_offset
    np.random.seed(seed)
    uninit_model = SGDClassifier(loss=loss, penalty="l2", max_iter = 1, warm_start=True, fit_intercept=True, random_state = seed, learning_rate='constant', eta0=lr_local)
    model, coefs, intercepts = init_model(uninit_model, dataset, num_clients, seed)

    c = {
        "coef" : np.zeros_like(model.coef_),
        "inter" : np.zeros_like(model.intercept_)
    }
    ci = np.array([c.copy()] * num_clients)
    old_ci = np.array([c.copy()] * num_clients)
    c_base = c.copy()
    ci_base = np.copy(ci)

    for round in range(num_global_rounds):
        #c = c_base.copy()
        #ci = np.copy(ci_base)
        for i in range(num_clients):
            old_ci[i] = ci[i].copy()
        task_list = np.empty(num_clients, dtype=object)

        print("starting round", round)
        ### request task from clients
        for i, org_id in enumerate (ids[0:num_clients]):
            round_task = client.post_task(
                input_= {
                    'method' : 'train_and_test',
                    'kwargs' : {
                        'model' : model,
                        'nb_parameters' : { 
                                'coef' : coefs[A_alt[i,:]],
                                'inter' : intercepts[A_alt[i,:]]
                        },
                        'classes' : model.classes_,
                        'use_scaffold': use_scaffold,
                        'use_dgd' : use_dgd,
                        'c' : c,
                        'ci' : ci[i],
                        'num_local_rounds' : num_local_epochs,
                        'num_local_batches' : num_local_batches
                        }
                },
                name =  "SVM" + ", round " + str(round),
                image = "sgarst/federated-learning:fedLin7",
                organization_ids=[org_id],
                collaboration_id= 1
            )
            task_list[i] =  round_task
        
        finished = False
        dataset_sizes = np.empty(num_clients, dtype = object)
        solved_tasks = []
        while (finished == False):
            for task_i, task in enumerate(task_list):
                result = client.get_results(task_id = task.get("id"))
                if not (None in [result[0]["result"]]):
                #print(result[0,0])
                    if not (task_i in solved_tasks):
                        #print(result)
                        res = (np.load(BytesIO(result[0]["result"]),allow_pickle=True))
                        accuracies[task_i, round] = res[0]
                        coefs[task_i,: ,:] = res[1]
                        intercepts[task_i,:] = res[2]
                        ci[task_i] = res[3]
                        dataset_sizes[task_i] = res[4]
                        solved_tasks.append(task_i)
            if len(solved_tasks) == num_clients:
                finished = True
            print("waiting")
            time.sleep(1)   
        
        
        if not use_dgd:
            print("averaging..")
            if use_scaffold:
                avg_coef, c = scaffold(dataset, None, model.coef_, coefs, c, old_ci, ci, lr_global, key = "coef")
                avg_intercept, c = scaffold(dataset, None, model.intercept_, intercepts, c, old_ci, ci, lr_global,key = "inter")
                c_log[round] = c['coef'].max()
                for i in range(num_clients):
                    ci_log[round, i] = ci[i]['coef'].max()
            else:
                avg_coef = average(coefs, dataset_sizes, None, dataset, None, use_sizes, False)
                avg_intercept = average(intercepts, dataset_sizes, None, dataset, None, use_sizes, False)


        


            model.coef_ = np.copy(avg_coef)
            model.intercept_ = np.copy(avg_intercept)
            global_accuracies[round] = model.score(X_test, y_test)
        #print(coefs[0].shape)
            prevmap.save_round(round, coefs, avg_coef, is_dict=False)
            newmap.save_round(round, coefs, avg_coef, is_dict=False)
        # 'global' test
        if round % 10 == 0:
            clear_database()

    if save_file:   
        if not use_dgd:
            prevmap.save_map(week+ save_str + "prevmap_seed" + str(seed) + ".npy")
            newmap.save_map(week + save_str + "nc_newmap_seed" + str(seed) + ".npy")
        ### save arrays to files
        with open (week + save_str + "_local_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, accuracies)
        with open (week + save_str + "_global_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, global_accuracies)



#rint(repr(accuracies))
print(repr(np.mean(accuracies, axis=1)))
print(repr(global_accuracies))
#print(np.mean(acc_results, axis=0))
print("final runtime", (time.time() - start_time)/60)
x = np.arange(num_global_rounds)

#prevmap.show_map()
#newmap.show_map()
cmap.show_map()
plt.plot(x, np.mean(accuracies, axis=0, keepdims=False))
plt.plot(x, global_accuracies)
#plt.plot(x, complete_test_results)
plt.show()

plt.plot(x, c_log)
plt.plot(x, ci_log)
plt.show()

