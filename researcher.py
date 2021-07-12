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

from io import BytesIO
from vantage6.client import Client
from helper_functions import average, get_datasets, get_config, scaffold, heatmap, get_save_str
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

criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'
lr_local = 5e-1
lr_global = 5e-1


ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

#dataset and booleans
dataset = 'MNIST_2class_IID'
week = "../datafiles/w11/"

save_file = True

#federated settings
num_global_rounds = 100
num_clients = 10
num_runs = 1
seed_offset = 0
num_clients = 10
lr = 0.5 

## parameter structures
avg_coef = np.zeros((1,784))
avg_intercept = np.zeros((1))
parameters = [avg_coef, avg_intercept]

coefs = np.zeros((num_clients, 784))
intercepts = np.zeros((num_clients))

# data structures to store results
accuracies = np.zeros((num_runs, num_clients, num_global_rounds))
prevmap = heatmap(num_clients, num_global_rounds)
newmap = heatmap(num_clients, num_global_rounds)

### main loop
for run in range(num_runs):
    seed = run + seed_offset
    torch.manual_seed(seed)
    #test model for global testing
    for round in range(num_global_rounds):

        print("starting round", round)
        ### request task from clients
        round_task = client.post_task(
            input_= {
                'method' : 'train_and_test',
                'kwargs' : {
                    'parameters' : parameters,
                    'seed' : seed
                    }
            },
            name =  "SVM" + ", round " + str(round),
            image = "sgarst/federated-learning:fedSVM",
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
        

        results = np.array(result)
        #print(np.array(results[0,1]))
        #print(results[:,1])
        accuracies[run, :, round] = np.array(results[:,0])
        coefs = np.array(results[:,1])
        intercepts = np.array(results[:,2])
        #print(coefs.shape)
        avg_coef = np.mean(coefs, axis=0)
        avg_intercept = np.mean(intercepts, axis=0)

        parameters = [avg_coef, avg_intercept]

        prevmap.save_round(round, coefs, avg_coef, is_dict=False)
        newmap.save_round(round, coefs, avg_coef, is_dict=False)
        # 'global' test
        
    prevmap.save_map(week + "SVM_IID_" + "prevmap_seed" + str(seed) + ".npy")
    newmap.save_map(week + "SVM_IID_" + "nc_newmap_seed" + str(seed) + ".npy")
    if save_file:
        ### save arrays to files
        with open (week + "SVM_IID" + "_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, accuracies)
'''
        with open (week + "SVM_IID_" + "global_seed"+ str(seed) + ".npy", 'wb') as f2:
            np.save(f2, complete_test_results)
'''

print(repr(accuracies))
print(repr(np.mean(accuracies, axis=1)))
#print(np.mean(acc_results, axis=0))
print("final runtime", (time.time() - start_time)/60)
x = np.arange(num_global_rounds)

prevmap.show_map()
newmap.show_map()

plt.plot(x, np.mean(accuracies, axis=1, keepdims=False)[0,:])
#plt.plot(x, complete_test_results)
plt.show()

    ### generate new model parameters
