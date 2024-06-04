import argparse
import torch
from torchvision.models import mobilenet_v3_small

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

    # Training
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 16
    metric_train = train_model(train_set, val_set, image_dir, model,
                               device, learning_rate=learning_rate, num_epochs=num_epochs, batch_size=batch_size)

    # Log Train results
    print(f"Metric fn : {metric_train}")

    # Testing
    results_df = test_model(test_set, image_dir, model, device)

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
