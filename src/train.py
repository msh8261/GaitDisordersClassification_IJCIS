# import glob
# import json
import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
# import yaml
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger

# from functools import partial
from config import config
from nn.classification import Classification as classifier
from src.dataset import PoseDataModule
from utils.utils import *

# import sys


num_keypoints = config.PARAMS["num_keypoints"]
lightning_logs_dir = config.PATH["lightning_logs"]
train_dataset_path = config.PATH["DATASET"]
num_classes_ = config.PARAMS["TOT_CLASSES"]
WINDOW_SIZE = config.PARAMS["WINDOW_SIZE"]
num_exercise = config.PARAMS["num_exercise"]


def print_torch_info():
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("Pytorch Lightening Version: ", pl.__version__)
    print("Number of Cuda on Device: ", torch.cuda.device_count())
    print("Cuda is available: ", torch.cuda.is_available())
    print("--------------------------------------------------")
    print("--------------------------------------------------")


def show_metrics(trainer, model_name):
    metrics = trainer.callback_metrics
    return metrics.items()


def test_x(X_train_path, info=""):
    file = open(X_train_path[0], "r")
    X = np.array([elem.split(",") for elem in file], dtype=np.float32)
    file.close()
    blocks = int(len(X) / (25 * 3))
    X_ = np.array(np.split(X, blocks))
    print("=========================================")
    print(info)
    print("=========================================")
    print(X_)
    print(X_.shape)


def test_y(y_test_path, info=""):
    file = open(y_test_path[0], "r")
    y = np.array(
        [elem for elem in [row.replace("  ", " ").strip().split(" ") for row in file]],
        dtype=np.int32,
    )
    file.close()
    print("=========================================")
    print(info)
    print("=========================================")
    print(y - 1)
    print(y.shape)


def save_scripted_module(model, model_name):
    save_model_path = (
        "models/" + model_name + "_scripted_model_" + str(num_classes_) + "class.pt"
    )
    scripted_model = torch.jit.script(model)
    # this script model is better way than trace it
    torch.jit.save(scripted_model, save_model_path)


def save_state_dict(model, model_name):
    optimizer = model.configure_optimizers()["optimizer"]
    save_path = "checkpoints/model_params/" + model_name + "_model.pt"
    torch.save((model.state_dict(), optimizer.state_dict()), save_path)


def do_training_validation(
    params, model_name, inputs_path_zipped, k, nums_folds, random_state
):
    num_features = config.PARAMS["features_size"]
    # default num_layers==1
    num_layers = 1
    # default dropout==0
    dropout = 0
    pl.seed_everything(random_state)
    parser = ArgumentParser()
    parent_parser = pl.Trainer.add_argparse_args(parser)
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--batch_size", type=int, default=params["batch_size"])
    parser.add_argument("--epochs", type=int, default=params["epochs"])
    parser.add_argument("--learning_rate", type=int, default=params["learning_rate"])
    parser.add_argument("--num_class", type=int, default=params["num_class"])
    parser.add_argument("--num_features", type=int, default=num_features)
    args, unknown = parser.parse_known_args()

    # initialize the model classifier
    model = classifier(
        model_name, num_features, num_features * 2, num_layers, args.num_class
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the modle on to the computation device and set to eval mode
    model.to(device).eval()

    data_module = PoseDataModule(
        inputs_path_zipped, args.batch_size, k, nums_folds, random_state
    )

    # save only the top 1 model based on val_loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint_" + model_name,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger("lightning_logs", name="model_run_" + model_name)

    # trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.epochs,
        deterministic=True,
        gpus=1,
        progress_bar_refresh_rate=1,
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=100),
            checkpoint_callback,
            lr_monitor,
        ],
    )

    tuner = pl.tuner.tuning.Tuner(trainer)
    new_batch_size = tuner.scale_batch_size(model, data_module, init_val=8)
    model.batch_size = new_batch_size
    lr_finder = tuner.lr_find(model, data_module)
    new_lr = lr_finder.suggestion()
    model.hparams.lr = new_lr

    trainer.fit(model, data_module)

    return model, trainer


if __name__ == "__main__":

    params = {}
    params["batch_size"] = WINDOW_SIZE * num_exercise
    params["epochs"] = 500
    params["learning_rate"] = 0.05
    params["num_class"] = num_classes_
    nums_folds = 10
    random_state = 21

    models_name = ["DHAT_LSTM", "tpcnn", "transformer"]

    X_train_path = os.path.join(train_dataset_path, "Xtrain.File")
    y_train_path = os.path.join(train_dataset_path, "ytrain.File")
    X_test_path = os.path.join(train_dataset_path, "Xtest.File")
    y_test_path = os.path.join(train_dataset_path, "ytest.File")

    inputs_path_zipped = [(X_train_path, X_test_path, y_train_path, y_test_path)]

    scores = []
    for i, model_name in enumerate(models_name):
        model_scores = []
        os.makedirs(lightning_logs_dir + "/" + model_name, exist_ok=True)
        for k in range(nums_folds):
            ## training
            model, trainer = do_training_validation(
                params, model_name, inputs_path_zipped, k, nums_folds, random_state
            )

            metrics_items = show_metrics(trainer, model_name)

            model_scores.append(metrics_items)

        scores.append(model_scores)
        with open(
            "results/"
            + str(models_name[i])
            + "_"
            + str(nums_folds)
            + "fold"
            + "_"
            + str(num_classes_)
            + "classes_"
            + str(WINDOW_SIZE * 3)
            + "_sequences"
            + "_results.txt",
            "w",
        ) as f:
            for j, score in enumerate(model_scores):
                for key, value in score:
                    if j == 0:
                        f.write("%s," % (key))
                f.write("\n")
                for key, value in score:
                    f.write("%s," % (value.cpu().detach().numpy()))
