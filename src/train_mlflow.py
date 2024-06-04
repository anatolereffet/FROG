import mlflow
import mlflow.pytorch
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import MSELoss
from utils.dataset import Dataset
from utils.metrics import metric_fn


def train(train_set, val_set, image_dir, model, device):
    train_set = Dataset(train_set, image_dir)
    val_set = Dataset(val_set, image_dir)

    params_train = {"batch_size": 1, "shuffle": True, "num_workers": 0}
    params_val = {"batch_size": 1, "shuffle": False, "num_workers": 0}

    training_generator = DataLoader(train_set, **params_train)
    validation_generator = DataLoader(val_set, **params_val)

    ###################   MODEL   #################
    if torch.cuda.is_available():
        print("\nCuda available")
        model.cuda()

    # Loss
    loss_fn = MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    print("\nTraining model ...")

    # MLflow: Start a new run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("batch_size", params_train["batch_size"])
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("num_epochs", 1)

        num_epochs = 1
        for n in range(num_epochs):
            print(f"Epoch {n}")
            for batch_idx, (X, y, gender, filename) in tqdm(
                enumerate(training_generator), total=len(training_generator)
            ):
                # Transfer to GPU
                X, y = X.to(device), y.to(device)
                y = torch.reshape(y, (len(y), 1))
                y_pred = model(X)
                loss = loss_fn(y_pred, y)

                if loss.isnan():
                    print(filename)
                    print("label", y)
                    print("y_pred", y_pred)
                    break

                if batch_idx % 200 == 0:
                    print(loss)
                    # Log the loss as a metric
                    mlflow.log_metric(
                        "loss", loss.item(), step=batch_idx + n * len(training_generator))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        results_list = []
        for batch_idx, (X, y, gender, filename) in tqdm(
            enumerate(validation_generator), total=len(validation_generator)
        ):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            for i in range(len(X)):
                results_list.append(
                    {"pred": float(y_pred[i]), "target": float(
                        y[i]), "gender": float(gender[i])}
                )
        results_df = pd.DataFrame(results_list)

        results_male = results_df.loc[results_df["gender"] > 0.5]
        results_female = results_df.loc[results_df["gender"] < 0.5]

        # Log the model
        mlflow.pytorch.log_model(model, "model")

        # Assuming metric_fn is defined and returns a metric dictionary
        metrics = metric_fn(results_male, results_female)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

    return metrics