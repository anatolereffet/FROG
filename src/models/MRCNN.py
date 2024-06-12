import torch
import torch.nn as nn
import pandas as pd
import mlflow

from utils.metrics import metric_fn


class MRCNN(nn.Module):
    def __init__(self, learning_rate=1e-3):
        super(MRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(40, 80, kernel_size=7, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(320, 1)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.train_results = []
        self.val_results = []

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
