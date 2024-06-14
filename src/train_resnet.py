import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from utils.dataset import Dataset
from utils.metrics import metric_fn
from utils.transform_loader import basic_transform, horizontal_transform, rotation_transform


def train(train_set, val_set, image_dir, device, model, **params):
    train_set_bas = Dataset(train_set, image_dir, transforms=basic_transform)
    train_set_hor = Dataset(train_set, image_dir,
                            transforms=horizontal_transform)
    train_set_rot = Dataset(train_set, image_dir,
                            transforms=rotation_transform)

    train_enhanced = ConcatDataset(
        [train_set_bas, train_set_hor, train_set_rot])

    val_set = Dataset(val_set, image_dir,
                      transforms=basic_transform, mode="val")

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

    training_generator = DataLoader(train_enhanced, **params_trainloader)
    validation_generator = DataLoader(val_set, **params_valloader)

    ###################   MODEL   #################
    learning_rate = params_train["learning_rate"]
    num_epochs = params_train["num_epochs"]
    batch_size = params_train["batch_size"]

    # Loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("\nTraining model ...")
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)

        best_val_metric = 5
        for n in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            print(f"Epoch {n+1}")

            for phase in ['train', 'val']:
                results_list = []
                if phase == 'train':
                    model.train()
                    image_datasets = training_generator
                else:
                    model.eval()
                    image_datasets = validation_generator

                running_loss = 0.0

                for _, (inputs, labels, gender, _) in tqdm(
                    enumerate(image_datasets), total=len(image_datasets)
                ):
                    inputs = inputs.to(device)
                    labels = labels.to(device).float()
                    labels = torch.reshape(labels, (len(labels), 1))

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        for i in range(len(inputs)):
                            results_list.append(
                                {
                                    "pred": float(outputs[i]),
                                    "target": float(labels[i]),
                                    "gender": float(gender[i]),
                                }
                            )

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(image_datasets)

                if phase == 'train':
                    train_loss = epoch_loss

                else:
                    val_loss = epoch_loss
                # Log the metrics
                results_df = pd.DataFrame(results_list)
                results_male = results_df.loc[results_df["gender"] > 0.5]
                results_female = results_df.loc[results_df["gender"] < 0.5]
                glob_metric, metric_male, metric_female = metric_fn(
                    results_male, results_female, detail=True
                )
                mlflow.log_metric(f"{phase}_metric_fn",
                                  glob_metric, step=n + 1)
                mlflow.log_metric(f"{phase}_metric_fn_male",
                                  metric_male, step=n + 1)
                mlflow.log_metric(f"{phase}_metric_fn_female",
                                  metric_female, step=n + 1)

            # Log the losses
            mlflow.log_metric("train_loss", train_loss, step=n)
            mlflow.log_metric("val_loss", val_loss, step=n)
        mlflow.pytorch.log_model(model, "model")
    return glob_metric
