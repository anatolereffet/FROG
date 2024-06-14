import argparse
import torch

from models.MTCNN import MTCNN
from models.MRCNN import MRCNN
from models.RESNET import RESNET

from utils.dataset import load_data, split_data
from train_resnet import train as train_model
from test import test as test_model


def main(parent_dir, runner, submission_ready, modelname):
    image_dir = f"{parent_dir}/crops_100K"
    train_set, test_set = load_data(parent_dir)

    train_set, test_set, val_set = split_data(
        train_set, test_set, runner=runner)

    print(f"Train set: {len(train_set)}")
    print(f"Validation set: {len(val_set)}")
    print(f"Test set: {len(test_set)}")

    if torch.cuda.is_available():
        print("\nCuda available")
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = "mps"

    if modelname == "MRCNN":
        model = MRCNN()

    if modelname == "ResNet":
        model = RESNET()

    else:
        model = MTCNN()

    # Training
    learning_rate = 0.00005
    num_epochs = 30
    batch_size = 32
    metric_train = train_model(
        train_set,
        val_set,
        image_dir,
        model,
        device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    # Log Train results
    print(f"Metric fn : {metric_train}")

    # Testing
    results_df = test_model(test_set, image_dir, device)

    submission_ready = True

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
        "-s", "--submission_ready", help="Dump results to csv in main directory", default="True"
    )

    parser.add_argument(
        "-m",
        "--modelname",
        help="Model to use for training",
        default="MTCNN",
    )
    args = parser.parse_args()
    main(args.parent_dir, args.runner, args.submission_ready, args.modelname)
