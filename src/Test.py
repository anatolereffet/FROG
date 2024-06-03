import pandas as pd
import torch
from tqdm import tqdm
from utils.dataset import Dataset_test


def main(df_test, image_dir , model, device):
    print("\nEvaluating test dataset...")
    test_set = Dataset_test(df_test, image_dir)
    params_test = {"batch_size": 8, "shuffle": False, "num_workers": 0}
    test_generator = torch.utils.data.DataLoader(test_set, **params_test)

    results_list = []
    for batch_idx, (X, filename) in tqdm(enumerate(test_generator), total=len(test_generator)):
        X = X.to(device)
        y_pred = model(X)
        for i in range(len(X)):
            results_list.append({"pred": float(y_pred[i])})
    test_df = pd.DataFrame(results_list)
    return test_df