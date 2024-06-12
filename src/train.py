import mlflow
import mlflow.pytorch
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from utils.dataset import Dataset
from utils.metrics import metric_fn


def train(train_set, val_set, image_dir, model, device, **params):
    train_set = Dataset(train_set, image_dir, mode="train")
    val_set = Dataset(val_set, image_dir, mode="val")

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

    training_generator = DataLoader(train_set, **params_trainloader)
    validation_generator = DataLoader(val_set, **params_valloader)

    ###################   MODEL   #################
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.to(device)

    learning_rate = params_train["learning_rate"]
    num_epochs = params_train["num_epochs"]
    batch_size = params_train["batch_size"]

    # Loss
    loss_fn = MSELoss()

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
            print(f"Epoch {n}")
            model.train()
            results_list = []
            train_loss = 0.0
            val_loss = 0.0
            for batch_idx, (X, y, gender, filename) in tqdm(enumerate(training_generator), total=len(training_generator)):
                X, y = X.to(device), y.to(device)
                y = torch.reshape(y, (len(y), 1))
                y_pred = model(X)
                loss = loss_fn(y_pred, y)

                if loss.isnan():
                    print(filename)
                    print("label", y)
                    print("y_pred", y_pred)
                    break

                for i in range(len(X)):
                    results_list.append(
                        {
                            "pred": float(y_pred[i]),
                            "target": float(y[i]),
                            "gender": float(gender[i]),
                        }
                    )

                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mlflow.log_metric(
                "train_loss", train_loss/len(training_generator), step=n+1)

            results_df = pd.DataFrame(results_list)
            results_male = results_df.loc[results_df["gender"] > 0.5]
            results_female = results_df.loc[results_df["gender"] < 0.5]
            glob_metric, metric_male, metric_female = metric_fn(
                results_male, results_female, detail=True
            )

            mlflow.log_metric("train_metric_fn", glob_metric, step=n + 1)
            mlflow.log_metric("train_metric_fn_male", metric_male, step=n + 1)
            mlflow.log_metric("train_metric_fn_female",
                              metric_female, step=n + 1)

            model.eval()
            with torch.no_grad():
                results_list = []
                for batch_idx, (X, y, gender, filename) in tqdm(enumerate(validation_generator), total=len(validation_generator)):
                    X, y = X.to(device), y.to(device)
                    y = torch.reshape(y, (len(y), 1))
                    y_pred = model(X)
                    vloss = loss_fn(y_pred, y)

                    if val_loss.isnan():
                        print(filename)
                        print("label", y)
                        print("y_pred", y_pred)
                        break

                    for i in range(len(X)):
                        results_list.append(
                            {
                                "pred": float(y_pred[i]),
                                "target": float(y[i]),
                                "gender": float(gender[i]),
                            }
                        )
                    val_loss += vloss.item()

                mlflow.log_metric(
                    "val_loss",
                    val_loss,
                    step=n + 1,
                )

                results_df = pd.DataFrame(results_list)
                results_male = results_df.loc[results_df["gender"] > 0.5]
                results_female = results_df.loc[results_df["gender"] < 0.5]

                glob_metric, metric_male, metric_female = metric_fn(
                    results_male, results_female, detail=True
                )
                if glob_metric < best_val_metric:
                    best_val_metric = glob_metric
                    torch.save(model.state_dict(),
                               f"./src/model_path/model50ep/epoch{n+1}.pth")

                mlflow.log_metric("val_metric_fn", glob_metric, step=n + 1)
                mlflow.log_metric("val_metric_fn_male",
                                  metric_male, step=n + 1)
                mlflow.log_metric("val_metric_fn_female",
                                  metric_female, step=n + 1)

        # Log the model
        mlflow.pytorch.log_model(model, "model")
    return glob_metric
