# import os

import numpy as np
import pytorch_lightning as pl
# import torch
# from torch.utils.data import WeightRandomSampler
from sklearn.model_selection import KFold
# from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset

import src.augmentation as aug
# from logging import exception
from config import config

WINDOW_SIZE = config.PARAMS["WINDOW_SIZE"]
num_exercise = config.PARAMS["num_exercise"]
num_augment = config.PARAMS["num_augment"]
add_augmentation = config.PARAMS["add_augmentation"]


def data_augmentation(data0, num=4):
    data0 = np.array(data0)
    blocks = int(len(data0) / (WINDOW_SIZE * num_exercise))
    data0 = np.array(np.split(data0, blocks))

    collect = []
    for i, data in enumerate(data0):
        data = data.reshape(1, data.shape[0], data.shape[1])
        x1 = aug.jitter(data)
        x2 = aug.scaling(data)
        x3 = aug.magnitude_warp(data)
        x4 = aug.time_warp(data)
        x5 = aug.rotation(data)
        x6 = aug.window_slice(data)

        if num == 0:
            X = data
        elif num == 1:
            X = np.vstack((data, x1))
        elif num == 2:
            X = np.vstack((data, x1, x2))
        elif num == 3:
            X = np.vstack((data, x1, x2, x3))
        elif num == 4:
            X = np.vstack((data, x1, x2, x3, x4))
        elif num == 5:
            X = np.vstack((data, x1, x2, x3, x4, x5))
        elif num == 6:
            X = np.vstack((data, x1, x2, x3, x4, x5, x6))
        else:
            assert print("INFO: please select a number between 0 and 6")

        collect.append(X)
    XX = np.vstack(collect)

    XX = XX.reshape(XX.shape[0] * XX.shape[1], XX.shape[2])

    return XX


def label_augmentation(target, num=4):
    y = np.array(target)

    collect = []
    for i, y in enumerate(y):
        if num == 0:
            yy = y
        elif num == 1:
            yy = np.vstack((y, y))
        elif num == 2:
            yy = np.vstack((y, y, y))
        elif num == 3:
            yy = np.vstack((y, y, y, y))
        elif num == 4:
            yy = np.vstack((y, y, y, y, y))
        elif num == 5:
            yy = np.vstack((y, y, y, y, y, y))
        elif num == 6:
            yy = np.vstack((y, y, y, y, y, y, y))
        else:
            assert print("INFO: please select a number between 0 and 6")

        collect.append(yy)
    yy = np.vstack(collect)

    return yy


class PoseDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PoseDataModule(pl.LightningDataModule):
    def __init__(self, data_path_zipped, batch_size, k, nums_folds, random_state):
        super().__init__()

        self.batch_size = batch_size
        self.k = k
        self.num_splits = nums_folds
        self.split_seed = random_state

        self.num_aug = 5

        X_train_path, X_test_path, y_train_path, y_test_path = list(
            zip(*data_path_zipped)
        )
        self.X_train_path = X_train_path[0]
        self.y_train_path = y_train_path[0]
        self.X_test_path = X_test_path[0]
        self.y_test_path = y_test_path[0]

    def load_X(self, X_path, mode):
        file = open(X_path, "r")

        X = np.array([elem.split(",") for elem in file], dtype=np.float32)
        file.close()

        blocks = int(len(X) / (WINDOW_SIZE * num_exercise))
        print(len(X), blocks)
        try:
            X_ = np.array(np.split(X, blocks))
        except:
            assert print("INFO: number of blocks is zero, please ckeck dataset.")

        print(X_.shape)
        return X_

    # Load the networks outputs
    def load_y(self, y_path, mode):
        file = open(y_path, "r")
        y = np.array(
            [
                elem
                for elem in [row.replace("  ", " ").strip().split(" ") for row in file]
            ],
            dtype=np.int32,
        )
        file.close()
        # for 0-based indexing
        y = y - 1

        return y

    def setup(self, stage=None):

        X_train = self.load_X(self.X_train_path, "train")
        y_train = self.load_y(self.y_train_path, "train")
        X_test = self.load_X(self.X_test_path, "test")
        y_test = self.load_y(self.y_test_path, "test")

        train_percent = 0.75
        test_percent = 0
        val_percent = 1 - (train_percent + test_percent)
        train_size = int(train_percent * len(X_train))
        val_size = int((len(X_train) - train_size) * val_percent)
        test_size = int(len(X_train) - (train_size + val_size))

        self.test_dataset = PoseDataset(X_test, y_test)

        # choose fold to train on
        kf = KFold(n_splits=self.num_splits, random_state=self.split_seed, shuffle=True)
        all_splits = [k for k in kf.split(X_train)]
        train_indexes, val_indexes = all_splits[self.k]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

        self.train_set = PoseDataset(X_train[train_indexes], y_train[train_indexes])
        self.val_set = PoseDataset(X_train[val_indexes], y_train[val_indexes])

    def train_dataloader(self):
        # train loader
        train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )
        return train_loader

    def val_dataloader(self):
        # validation loader
        val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        return val_loader

    def test_dataloader(self):
        # test loader
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
        return test_loader
