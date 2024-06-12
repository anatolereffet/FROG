import torch
import pandas as pd
import numpy as np
from PIL import Image

# from torchvision.transforms import v2
from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, df, image_dir, mode="train"):
        "Initialization"
        self.image_dir = image_dir
        self.df = df
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)
        self.mode = mode
        if mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=224, width=224),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.df)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        row = self.df.loc[index]
        filename = row["filename"]

        # Load data and get label
        img = Image.open(f"{self.image_dir}/{filename}")

        # X = self.transform(img)
        aug = self.transform(image=np.array(img))
        X = aug["image"]

        if self.mode in ("train", "val"):
            y = np.float32(row["FaceOcclusion"])
            gender = row["gender"]

            return X, y, gender, filename
        else:
            return X, filename


def load_data(parent_dir: str):
    """
    Simple data loading according to parent directory
    """
    train = pd.read_csv(f"{parent_dir}/listes_training/data_100K/train_100K.csv", delimiter=" ")
    test = pd.read_csv(f"{parent_dir}/listes_training/data_100K/test_students.csv", delimiter=" ")

    return train, test


def split_data(train, test, runner: bool = False, n_samples: Optional[int] = 100):
    """
    Data split

    Args:
        train : train set
        test : _description_
        runner (bool, optional): Real run or simple trial. Defaults to False.
        n_samples (Optional[int], optional): If it isn't a real run, number of samples to be used.
                                            Defaults to 100.

    Returns:
        _type_: _description_
    """
    n_split = 20000

    val = train.loc[:n_split]

    if not runner:
        print(f"Training on a sample of the data {n_samples}")
        n_split = n_samples
        test = test.loc[:n_split]
        train = train.loc[n_split : n_samples * 2]
    else:
        print("Training on the complete dataset")
        train = train.loc[n_split:]

    return train, test, val
