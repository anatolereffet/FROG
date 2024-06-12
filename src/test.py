import pandas as pd
import torch
from tqdm import tqdm
from utils.dataset import Dataset


def test(test_set, image_dir, model, device):
    print("\nTesting model ...")
    test_set = Dataset(test_set, image_dir, mode="test")
    params_test = {"batch_size": 8, "shuffle": False, "num_workers": 0}
    test_generator = torch.utils.data.DataLoader(test_set, **params_test)

    results_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, filename) in enumerate(test_generator):
            X = X.to(device)
            y_pred = model(X)
            for i in range(len(X)):
                results_list.append({"pred": float(y_pred[i])})
            progress = int((i + 1) / len(test_generator) * 100)
            print("\r[ {0}{1} ] {2}%".format("#" * progress,
                  " " * int(100 - progress), progress), end="",)

    test_df = pd.DataFrame(results_list)
    return test_df
