import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, df, image_dir, train=True):
        "Initialization"
        self.image_dir = image_dir
        self.df = df
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)
        if train:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.ToTensor()

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
        y = row["FaceOcclusion"]
        gender = row["gender"]

        X = self.transform(img)
        y = np.float32(y)

        return X, y, gender, filename
