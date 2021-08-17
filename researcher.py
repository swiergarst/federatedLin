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

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from sklearn.linear_model import SGDClassifier
from io import BytesIO
from vantage6.client import Client
from helper_functions import average, get_datasets, get_full_dataset,get_config, scaffold, heatmap, get_save_str
start_time = time.time()
### connect to server


print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = "/home/swier/.local/share/vantage6/node/privkey_testOrg0.pem"
client.setup_encryption(privkey)




#organizations = client.get_organizations_in_my_collaboration()
#org_ids = [organization["id"] for organization in organizations]




### parameter settings

#torch

lr_local = 5e-1
lr_global = 5e-1


ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

#dataset and booleans
dataset = 'MNIST_2class' #either MNIST_2class or MNIST_4class
week = "../datafiles/w14/"
classifier = "SVM" #either SVM or LR

save_file = True
class_imbalance = False
sample_imbalance = False
use_scaffold = False
use_sizes = False

save_str = get_save_str(dataset, classifier, class_imbalance, sample_imbalance, use_scaffold, use_sizes, lr_local, 1, 1)

#federated settings
num_global_rounds = 100
num_clients = 10
num_runs = 4
seed_offset = 0
num_clients = 10

if classifier == "SVM":
    loss = "hinge"
elif classifier == "LR":
    loss = "log"
else:
    raise(Exception("unkown classifier provided"))


# data structures to store results
accuracies = np.zeros((num_runs, num_clients, num_global_rounds))
global_accuracies = np.zeros((num_runs, num_global_rounds))
prevmap = heatmap(num_clients, num_global_rounds)
newmap = heatmap(num_clients, num_global_rounds)

datasets = get_datasets(dataset)
X_test, y_test = get_full_dataset(datasets, "FNN")
X_test = X_test.numpy()
y_test = y_test.numpy()

### main loop
for run in range(num_runs):
    seed = run + seed_offset
    np.random.seed(seed)
    model = SGDClassifier(loss=loss, penalty="l2", max_iter = 1, warm_start=True, fit_intercept=True, random_state = seed, learning_rate='constant', eta0=lr_local)
    
    if dataset == "MNIST_4class":
        coefs = np.zeros((num_clients, 4, 784))
        avg_coef = np.zeros((4,784))
        avg_intercept = np.zeros((4))
        intercepts = np.zeros((num_clients, 4))
        model.coef_ = np.random.rand(4, 784)
        model.intercept_ = np.random.rand(4)
        classes = np.array([0,1,2,3])
        model.classes_ = classes
    else:
        avg_coef = np.zeros((1,784))
        coefs = np.zeros((num_clients, 784))
        avg_intercept = np.zeros((1))
        intercepts = np.zeros((num_clients))
        model.coef_ = np.random.rand(1, 784)
        model.intercept_ = np.random.rand(1,1)
        classes = np.array([0,1])
        model.classes_ = classes
    c = [np.zeros_like(model.coef_), np.zeros_like(model.intercept_)]
    ci = np.array([np.zeros_like(model.coef_), np.zeros_like(model.intercept_)] * num_clients) 

    for round in range(num_global_rounds):

        print("starting round", round)
        ### request task from clients
        round_task = client.post_task(
            input_= {
                'method' : 'train_and_test',
                'kwargs' : {
                    'model' : model,
                    'classes' : classes,
                    'use_scaffold': use_scaffold,
                    'c' : c,
                    'ci' : ci
                    }
            },
            name =  "SVM" + ", round " + str(round),
            image = "sgarst/federated-learning:fedSVM3",
            organization_ids=ids,
            collaboration_id= 1
        )
        
        #print(round_task)
        info("Waiting for results")
        res = client.get_results(task_id=round_task.get("id"))
        attempts=1
        #print(res)
        while(None in [res[i]["result"] for i in range(num_clients)]  and attempts < 20):
            print("waiting...")
            time.sleep(1)
            res = client.get_results(task_id=round_task.get("id"))
            attempts += 1

        info("Obtaining results")
        #result  = client.get_results(task_id=task.get("id"))
        result = []
        for i in range(num_clients):
            result.append(np.load(BytesIO(res[i]["result"]),allow_pickle=True))
        

        results = np.array(result, dtype=object)


        accuracies[run, :, round] = results[:,0]
        
        #print(results[0,1])

        if dataset == "MNIST_4class":
            for c in range(num_clients):
                coefs[c,:,:] = results[c,1]
                intercepts[c,:] = results[c,2]
        else:
            for c in range(num_clients):
                coefs[c,:] = results[c,1]
                intercepts[c] = results[c,2]
        

        if use_scaffold:
            avg_coef = scaffold(coefs, axis=0, keepdims=False)
            avg_intercept = scaffold(intercepts, axis=0, keepdims=False)
        else:
            avg_coef = average()
            avg_intercept = average()
        #sys.exit()
        #coefs = results[:,1]
        #intercepts = results[:,2]
        


        model.coef_ = avg_coef
        model.intercept_ = avg_intercept
        global_accuracies[run, round] = model.score(X_test, y_test)
        #print(coefs[0].shape)
        prevmap.save_round(round, coefs, avg_coef, is_dict=False)
        newmap.save_round(round, coefs, avg_coef, is_dict=False)
        # 'global' test
        

    if save_file:   
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

prevmap.show_map()
newmap.show_map()

plt.plot(x, np.mean(accuracies, axis=1, keepdims=False)[0,:])
plt.plot(x, global_accuracies[0,:])
#plt.plot(x, complete_test_results)
plt.show()

    ### generate new model parameters
