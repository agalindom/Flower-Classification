import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from torch import nn
from torch import optim
import numpy as np
from PIL import Image
import seaborn as sns
import argparse
import m_func


arg = argparse.ArgumentParser(description = 'Train_a_model')
arg.add_argument('data_folder', nargs = '*', action = 'store', default = 'aipnd_project/flowers/')
arg.add_argument('--nn_directory', dest = 'nn_directory', action = 'store', default = 'aipnd_project/chkpoint.pth')
arg.add_argument('--learning_rate', dest = 'learning_rate', action = 'store', default = 0.001)
arg.add_argument('--dropout', dest = 'dropout', action = 'store', default = 0.2)
arg.add_argument('--epochs', dest = 'epochs', action = 'store', default = 6)

pa = arg.parse_args()
directory = pa.data_folder
dropout = pa.dropout
learning_rate = pa.learning_rate
epochs = pa.epochs

trainloader, validloader, testloader, train_data = m_func.loader()
m_func.train_network(epochs = epochs, data = trainloader, eval_data = validloader, mapping_data = train_data)




                              