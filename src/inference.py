import argparse
import torch

from models.network import MTCNN

from utils.dataset import load_data, split_data
from test import test as test_model


def main(parent_dir, model_path):
    image_dir = f"{parent_dir}/crops_100K"
    train_set, test_set = load_data(parent_dir)

    _, test_set, _ = split_data(train_set, test_set, runner=True)

    print("Inference starting")

    model = MTCNN()

    device = "mps"
    torch.backends.cudnn.benchmark = True
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Testing
    results_df = test_model(test_set, image_dir, model, device)

    results_df.to_csv("results_tmp23.csv", header=None, index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pdd",
        "--parent_dir",
        help="Path to the folder data/ holding crops_100K or liste_training",
        default="./data",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        help="Model path for inference",
        default="./",
    )
    args = parser.parse_args()
    main(args.parent_dir, args.model_path)
