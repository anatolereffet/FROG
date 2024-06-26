import pandas as pd
import torch
from tqdm import tqdm
from utils.dataset import Dataset
from utils.transform_loader import basic_transform


def test(test_set, image_dir, model, device):
    print("\nTesting model ...")
    test_set = Dataset(test_set, image_dir, transforms=basic_transform, mode="test")
    params_test = {"batch_size": 8, "shuffle": False, "num_workers": 0}
    test_generator = torch.utils.data.DataLoader(test_set, **params_test)

    results_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, filename) in tqdm(enumerate(test_generator), total=len(test_generator)):
            X = X.to(device)
            y_pred = model(X)
            for i in range(len(X)):
                results_list.append({"pred": float(y_pred[i])})

    test_df = pd.DataFrame(results_list)
    return test_df
