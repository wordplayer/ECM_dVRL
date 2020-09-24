import gym
import math
import time
import random
import ECM_dVRL_v01
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from mpl_toolkits import mplot3d

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class deepQnet:
    def __init__(self, env, action_space):
        self.env = env
        self.action_space = action_space
    
    