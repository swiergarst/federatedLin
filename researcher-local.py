from numpy.lib.npyio import save
from vantage6.tools.mock_client import ClientMockProtocol
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from helper_functions import get_datasets, get_full_dataset, heatmap
from sklearn.linear_model import SGDClassifier

dataset = "MNIST_4class"
### connect to server
datasets = get_datasets(dataset)
#datasets.remove("/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client9.csv")
client = ClientMockProtocol(
    datasets= datasets,
    module="v6_svm_py"
### connect to server
)
organizations = client.get_organizations_in_my_collaboration()
org_ids = [organization["id"] for organization in organizations]

save_file  = False


lr = 0.5
num_runs = 1
num_global_rounds = 20
avg_coef = np.zeros((1,784))
avg_intercept = np.zeros((1))
parameters = [avg_coef, avg_intercept]
num_clients = 10
seed_offset = 0


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
    model = SGDClassifier(loss="hinge", penalty="l2", max_iter = 1, warm_start=True, fit_intercept=True, random_state = seed)
    
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
    
    map = heatmap(num_clients, num_global_rounds )
    for round in range(num_global_rounds):
        round_task = client.create_new_task(
            input_= {
                'method' : 'train_and_test',
                'kwargs' : {
                    'model' : model,
                    'classes': classes
                    }
            },
            organization_ids=org_ids
        )
        ## aggregate responses
        results = np.array(client.get_results(round_task.get("id")))
        accuracies[run, :,round] = results[:, 0]
        coefs_tmp = results[:,1]
        intercepts_tmp = results[:,2]
        for i, coef  in enumerate(coefs_tmp):
            coefs[i,:] = coef
        #print(intercepts)
        for i, intercept in enumerate(intercepts_tmp):
            #print(intercept[0])
            intercepts[i] = intercept
        #print(intercepts)
        #sys.exit()
        coef_log_l[run,round,:] = coefs[:, 345]
        intercept_agg = 0
        coef_agg = np.zeros((784))
        #for i in range(num_clients):
        #    intercept_agg += intercepts[i] - avg_intercept
        #    coef_agg +=  intercepts[i] - avg_intercept

        #avg_intercept = avg_intercept + (lr / num_clients) * intercept_agg
        #avg_coef = avg_coef + (lr / num_clients) * coef_agg
        global_accuracies[run, round] = model.score(X_test, y_test)

        avg_coef = np.mean(coefs, axis=0, keepdims=True)
        avg_intercept = np.mean(intercepts, axis=0, keepdims=True)
        model.coef_ = avg_coef
        model.intercept_ = avg_intercept

        coef_log_g[run, round] = avg_coef[0,345]
        print("in main: ", avg_coef)
        map.save_round(round, coefs, avg_coef, is_dict=False)
        parameters = [avg_coef, avg_intercept]
    if save_file:
        map.save_map("../w10/simulated_svm_avg_no9_seed" + str(seed) + "map.npy")

if save_file:
    ### save arrays to files
    with open ("../w10/simulated_svm_iid_avg_no9_seed" + str(seed)+ ".npy", 'wb') as f:
        np.save(f, accuracies)

print(repr(coef_log_l))
print(repr(coef_log_g))

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