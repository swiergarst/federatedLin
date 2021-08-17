from vantage6.client import Client
import numpy as np



print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = "/home/swier/.local/share/vantage6/node/privkey_testOrg0.pem"
client.setup_encryption(privkey)


in_array = np.array([30,31,32,33,34,35,36,37,38,39])

ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

