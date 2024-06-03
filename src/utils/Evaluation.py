import numpy as np

def error_fn(df):
    pred = df.loc[:, "pred"]
    ground_truth = df.loc[:, "target"]
    weight = 1 / 30 + ground_truth
    return np.sum(((pred - ground_truth) ** 2) * weight, axis=0) / np.sum(weight, axis=0)


def metric_fn(female, male):
    err_male = error_fn(male)
    err_female = error_fn(female)
    return (err_male + err_female) / 2 + abs(err_male - err_female)