import pandas as pd
import torch
import torchvision
import Train
import Test
import sys

def main(name_file= "Data_Challenge.csv"):
    ##### DATASET #####
    df_train = pd.read_csv("../data/listes_training/data_100K/train_100K.csv", delimiter=" ")
    df_test = pd.read_csv("../data/listes_training/data_100K/test_students.csv", delimiter=" ")
    image_dir = "../data/crops_100K"

    #Remove na
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    #split train in val and train
    df_val = df_train.loc[:100].reset_index()
    df_train = df_train.loc[100:200].reset_index()
    df_test = df_test.loc[:100]
    

    ##### SET MODEL #####
    model = torchvision.models.mobilenet_v3_small(num_classes=1)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True


    ##### TRAINING ##### 
    Train.main(df_train, df_test, image_dir, df_val, model, device)


    ##### TEST #####
    test_df = Test.main(df_test, image_dir, model, device)

    #File for submission
    #test_df.to_csv(name_file, header=None, index=None)

if __name__ == "__main__":
    if len(sys.argv) > 1:  # Checks if at least one argument is provided
        main(sys.argv[1])  # Passes the first argument to the main function
    else:
        print("File name not provided.")
        main()