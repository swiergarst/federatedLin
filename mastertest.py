from vantage6.client import Client
import numpy as np
import time
from io import BytesIO

print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = "/home/swier/.local/share/vantage6/node/privkey_testOrg0.pem"
client.setup_encryption(privkey)

num_clients = 10
in_array = np.array([30,31,32,33,34,35,36,37,38,39])

ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

task = client.post_task(
    input_ = {
        'master': 1,
        'method': 'master',
        'kwargs': {
            'id_array' : ids,
            'input_array' : in_array
        }
    },
    name = "mastertest",
    image = "sgarst/federated-learning:mastertest",
    organization_ids=ids,
    collaboration_id=1
)

#info("Waiting for results")
res = client.get_results(task_id=task.get("id"))
attempts=1
#print(res)
while(None in [res[i]["result"] for i in range(num_clients)]  and attempts < 20):
    print("waiting...")
    time.sleep(1)
    res = client.get_results(task_id=task.get("id"))
    attempts += 1

#  info("Obtaining results")
result = []
for i in range(num_clients):
    result.append(np.load(BytesIO(res[i]["result"]),allow_pickle=True))

print(result)