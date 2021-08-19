from numpy.lib.npyio import save
from vantage6.tools.mock_client import ClientMockProtocol
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from helper_functions import get_datasets, get_full_dataset, heatmap, average, scaffold
from sklearn.linear_model import SGDClassifier
import time
dataset = "MNIST_2class"


save_file  = False
class_imbalance = False
sample_imbalance = False
use_scaffold = True
use_sizes = False
### connect to server
datasets = get_datasets(dataset, class_imbalance, sample_imbalance)

#datasets.remove("/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client9.csv")
client = ClientMockProtocol(
    datasets= datasets,
    module="v6_svm_py"
### connect to server
)
organizations = client.get_organizations_in_my_collaboration()
org_ids = [organization["id"] for organization in organizations]



lr_local = 0.5
lr_global = 0.5

num_runs = 1
num_global_rounds = 5
num_clients = 10
seed_offset = 0


avg_coef = np.zeros((1,784))
avg_intercept = np.zeros((1))
parameters = [avg_coef, avg_intercept]
accuracies = np.zeros((num_runs, num_clients, num_global_rounds))
global_accuracies = np.zeros((num_runs, num_global_rounds))
coefs = np.zeros((num_clients, 784))
intercepts = np.zeros((num_clients))

coef_log_g =np.zeros((num_runs, num_global_rounds))
coef_log_l = np.zeros((num_runs, num_global_rounds, num_clients))

X_test, y_test = get_full_dataset(datasets, "FNN")
X_test = X_test.numpy()
y_test = y_test.numpy()

for run in range(num_runs):
    seed = run + seed_offset
    np.random.seed(seed)
    model = SGDClassifier(loss="hinge", penalty="l2", max_iter = 1, warm_start=True, fit_intercept=True, learning_rate="constant", eta0=lr_local, random_state = seed)
    
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
        coefs = np.zeros((num_clients,1, 784))
        avg_intercept = np.zeros((1))
        intercepts = np.zeros((num_clients,1))
        model.coef_ = np.random.rand(1, 784)
        model.intercept_ = np.random.rand(1,1)
        classes = np.array([0,1])
        model.classes_ = classes
    
    c = {
        "coef" : np.zeros_like(model.coef_),
        "inter" : np.zeros_like(model.intercept_)
    }
    ci = np.array([c.copy()] * num_clients)
    map = heatmap(num_clients, num_global_rounds )
    for round in range(num_global_rounds):
        old_ci = np.copy(ci)
        task_list = np.empty(num_clients, dtype=object)
        
        for i, org_id in enumerate (org_ids):
            round_task = client.create_new_task(
                input_= {
                    'method' : 'train_and_test',
                    'kwargs' : {
                        'model' : model,
                        'classes': classes,
                        'use_scaffold': use_scaffold,
                        'c' : c,
                        'ci' : ci[i]
                        }
                },
                organization_ids=[org_id]                
            )
            task_list[i] =  round_task
       
        finished = False
        #coefs_tmp = np.empty(num_clients, dtype=object)
        #inter_tmp = np.empty(num_clients, dtype=object)
        dataset_sizes = np.empty(num_clients, dtype = object)
        solved_tasks = []
        while (finished == False):
            #new_task_list = np.copy(task_list)

            for task_i, task in enumerate(task_list):
                result = client.get_results(task_id = task.get("id"))
                #print(result[0][0])
                #sys.exit()
                if not (None in [result[0]]): #["result"]]):
                #print(result[0,0])
                    if not (task_i in solved_tasks):
                        #print(result)
                        accuracies[run, task_i, round] = result[0][0]
                        coefs[task_i,: ,:] = result[0][1]
                        intercepts[task_i,:] = result[0][2]
                        ci[task_i] = result[0][3]
                        dataset_sizes[task_i] = result[0][4]
                        solved_tasks.append(task_i)
            if len(solved_tasks) == num_clients:
                finished = True
            print("waiting")
            time.sleep(1)   


       
       
        ## aggregate responses
        

        if use_scaffold:
            avg_coef, c = scaffold(dataset, None, model.coef_, coefs, c, old_ci, ci, lr_global, key = "coef")
            avg_intercept, c = scaffold(dataset, None, model.intercept_, intercepts, c, old_ci, ci, lr_global,key = "inter")
        else:
            avg_coef = average(coefs, dataset_sizes, None, dataset, None, use_sizes, False)
            avg_intercept = average(intercepts, dataset_sizes, None, dataset, None, use_sizes, False)

        model.coef_ = np.copy(avg_coef)
        model.intercept_ = np.copy(avg_intercept)


        global_accuracies[run, round] = model.score(X_test, y_test)

        coef_log_g[run, round] = avg_coef[0,345]
        #print("in main: ", avg_coef)
        map.save_round(round, coefs, avg_coef, is_dict=False)
        #parameters = [avg_coef, avg_intercept]
    if save_file:
        map.save_map("../w10/simulated_svm_avg_no9_seed" + str(seed) + "map.npy")

if save_file:
    ### save arrays to files
    with open ("../w10/simulated_svm_iid_avg_no9_seed" + str(seed)+ ".npy", 'wb') as f:
        np.save(f, accuracies)

#print(repr(coef_log_l))
#print(repr(coef_log_g))

x = np.arange(num_global_rounds)

#print(accuracies)
#plt.plot(x, coef_log_l[1,:,:].T)
#plt.plot(x, coef_log_g[1,:])
#plt.show()
#print(np.mean(accuracies, axis=1))
plt.plot(np.arange(num_global_rounds), np.mean(accuracies, axis = 1)[0,:])

plt.plot(np.arange(num_global_rounds), global_accuracies[0,:])
plt.show()
#map.show_map("SVM classifier, IID datasets")
#map.save_map("../w10/simulated_svm_average_seed" + str(seed) + "map.npy")