import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from utils.dataset import Dataset
from utils.Evaluation import error_fn, metric_fn

def main(df_train, df_test, image_dir, df_val, model, device):
    #check that all is read OK
    print("Check reading dataset...")
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train)):
        try:
            filename = df_train.loc[idx, "filename"]
            img2display = Image.open(f"{image_dir}/{filename}")
        except ValueError as e:
            print(idx, e)

    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        try:
            filename = df_test.loc[idx, "filename"]
            img2display = Image.open(f"{image_dir}/{filename}")
        except ValueError as e:
            print(idx, e)


    #Make dataset and Dataloader
    training_set = Dataset(df_train, image_dir)
    validation_set = Dataset(df_val, image_dir)

    params_train = {"batch_size": 1, "shuffle": True, "num_workers": 0}

    params_val = {"batch_size": 1, "shuffle": False, "num_workers": 0}
    
    training_generator = torch.utils.data.DataLoader(training_set, **params_train)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)


    ###################   MODEL   #################
    if torch.cuda.is_available():
        print("\nCuda available")
        model.cuda()

    #Loss
    
    loss_fn = nn.MSELoss()
    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # Train
    print("\nTraining model ...")

    # Fit
    num_epochs = 1

    for n in range(num_epochs):
        print(f"Epoch {n}")
        for batch_idx, (X, y, gender, filename) in tqdm(
            enumerate(training_generator), total=len(training_generator)
        ):
            # Transfer to GPU
            X, y = X.to(device), y.to(device)
            y = torch.reshape(y, (len(y), 1))
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            if loss.isnan():
                print(filename)
                print("label", y)
                print("y_pred", y_pred)
                break

            if batch_idx % 200 == 0:
                print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    #Evaluate
    results_list = []
    for batch_idx, (X, y, gender, filename) in tqdm(
        enumerate(validation_generator), total=len(validation_generator)
    ):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        for i in range(len(X)):
            results_list.append(
                {"pred": float(y_pred[i]), "target": float(y[i]), "gender": float(gender[i])}
            )
    results_df = pd.DataFrame(results_list)

    results_male = results_df.loc[results_df["gender"] > 0.5]
    results_female = results_df.loc[results_df["gender"] < 0.5]

    print("Metric fn :", metric_fn(results_male, results_female))

if __name__ == "__main__":
    main()