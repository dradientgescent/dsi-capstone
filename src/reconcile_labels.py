import os

import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def get_lst_images(file_path):
    """
    Reads in all files from file path into a list.

    INPUT
        file_path: specified file path containing the images.

    OUTPUT
        List of image strings
    """
    return [i for i in os.listdir(file_path) if i != '.DS_Store']


if __name__ == '__main__':
    df_train_labels = pd.read_csv("../labels/trainLabels_master.csv")

    X = np.array(df_train_labels.index).reshape(-1,1)
    Y = df_train_labels["level"].values
    
    """
    Correct class imbalance, under sampling
    """
    sample = RandomUnderSampler()
    X_resampled, Y_resampled = sample.fit_sample(X, Y)

    df_resampled = pd.DataFrame()
    df_resampled["image_idx"] = X_resampled[:,0]
    df_resampled["level"] = Y_resampled

    df = df_train_labels[df_train_labels.index.isin(df_resampled.image_idx)]

    df = df.reset_index(drop=True)

    df.to_csv("trainLabels_DEV.csv", index=False, header=True)

