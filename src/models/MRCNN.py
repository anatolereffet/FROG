import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pandas as pd

from utils.metrics import metric_fn


class MRCNN(pl.LightningModule):
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

    def training_step(self, batch, batch_idx):
        x, y, gender, _ = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.train_results.extend([
            {"pred": float(y_hat[i]), "target": float(
                y[i]), "gender": float(gender[i])}
            for i in range(len(x))
        ])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, gender, _ = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.val_results.extend([
            {"pred": float(y_hat[i]), "target": float(
                y[i]), "gender": float(gender[i])}
            for i in range(len(x))
        ])
        self.logger.log('val_loss', loss)
        return loss

    def on_train_epoch_end(self):
        results_df = pd.DataFrame(self.train_results)
        results_male = results_df.loc[results_df["gender"] > 0.5]
        results_female = results_df.loc[results_df["gender"] < 0.5]
        glob_metric, metric_male, metric_female = metric_fn(
            results_male, results_female, detail=True)
        self.logger.experiment.log_metric('train_metric_fn', glob_metric)
        self.logger.experiment.log_metric('train_metric_fn_male', metric_male)
        self.logger.experiment.log_metric(
            'train_metric_fn_female', metric_female)
        self.train_results = []  # reset for next epoch

    def on_validation_epoch_end(self):
        results_df = pd.DataFrame(self.val_results)
        results_male = results_df.loc[results_df["gender"] > 0.5]
        results_female = results_df.loc[results_df["gender"] < 0.5]
        glob_metric, metric_male, metric_female = metric_fn(
            results_male, results_female, detail=True)
        self.logger.experiment.log_metric('val_metric_fn', glob_metric)
        self.logger.experiment.log_metric('val_metric_fn_male', metric_male)
        self.logger.experiment.log_metric(
            'val_metric_fn_female', metric_female)
        self.val_results = []  # reset for next epoch

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
