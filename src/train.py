import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader
from utils.dataset import Dataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def train(train_set, val_set, image_dir, model, **params):
    train_set = Dataset(train_set, image_dir)
    val_set = Dataset(val_set, image_dir)

    default_params = {"learning_rate": 0.001,
                      "num_epochs": 10, "batch_size": 5}
    params_train = {**default_params, **params}

    params_trainloader = {
        "batch_size": params_train["batch_size"],
        "shuffle": True,
        "num_workers": 0,
    }

    params_valloader = {
        "batch_size": params_train["batch_size"],
        "shuffle": False,
        "num_workers": 0,
    }

    train_loader = DataLoader(train_set, **params_trainloader)
    val_loader = DataLoader(val_set, **params_valloader)

    learning_rate = params_train["learning_rate"]
    num_epochs = params_train["num_epochs"]
    batch_size = params_train["batch_size"]

    # Initialize MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name="FaceOcclusionDetection", tracking_uri="http://localhost:5000")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=10,
        logger=mlflow_logger,
        callbacks=[ModelCheckpoint(monitor='val_loss')]
    )

    # Train the model
    with mlflow.start_run() as run:
        mlflow_logger.log_hyperparams(model.hparams)
        trainer.fit(model, train_loader, val_loader)
        mlflow.pytorch.log_model(model, "models")
