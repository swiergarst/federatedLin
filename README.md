# Federated implementations of linear classifiers

This implementation uses the [vantage6](https://docs.vantage6.ai/) framework. Instructions on how to install vantage6 can be found [here](https://docs.vantage6.ai/installation/what-to-install)
Furthermore, both [SKLearn](https://scikit-learn.org/stable/) and [Pytorch](https://pytorch.org/) are required to run.

The main file to be run is `researcher.py. before trying to run this file, ensure the following things:

- private key location (`privkey = "path/to/private_key.pem"`)
- data file save location (`week = "path/to/folder"`)

Furthermore, some settings can be changed in order execute different experiments (e.g. different datasets)
