import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from engine.base_trainer import BaseTrainer
from tqdm import tqdm

class TaskTrainer(BaseTrainer)