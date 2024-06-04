import argparse
import torch
import mlflow
from torchvision.models import mobilenet_v3_small
from torch import nn
from torchmetrics import Accuracy

from utils.dataset import load_data, split_data
from train_mlflow import train as train_model
from test import test as test_model


def main(parent_dir, runner, submission_ready):
    image_dir = f"{parent_dir}/crops_100K"
    train_set, test_set = load_data(parent_dir)

    train_set, test_set, val_set = split_data(
        train_set, test_set, runner=runner)

    model = mobilenet_v3_small(num_classes=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    epochs = 3
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    batch_size = 1

    params_train = {"batch_size": batch_size,
                    "shuffle": True, "num_workers": 0}

    params_val = {"batch_size": batch_size,
                  "shuffle": False, "num_workers": 0}

    training_generator = torch.utils.data.DataLoader(train_set, **params_train)
    validation_generator = torch.utils.data.DataLoader(val_set, **params_val)

    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run() as run:

        params = {
            "epochs": epochs,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "loss_function": loss_fn.__class__.__name__,
            "optimizer": "SGD",
        }
    # Log training parameters.
        mlflow.log_params(params)

        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_model(train_dataloader, model, loss_fn,
                        metric_fn, optimizer, epoch=t)
            evaluate(test_dataloader, model, loss_fn, metric_fn, epoch=0)

        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, "model")

    if submission_ready:
        results_df.to_csv("results.csv", header=None, index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pdd",
        "--parent_dir",
        help="Path to the folder data/ holding crops_100K or liste_training",
        default="./data",
    )
    parser.add_argument(
        "-r",
        "--runner",
        help="Train on the real data split",
        default="False",
    )
    parser.add_argument(
        "-s", "--submission_ready", help="Dump results to csv in main directory", default="False"
    )
    args = parser.parse_args()
    main(args.parent_dir, args.runner, args.submission_ready)

##### FIRST LAUNCH mlflow ui in cli ##########
