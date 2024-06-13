import os

import numpy as np
import torch
import torch.nn.functional as F
# from sklearn.metrics import (accuracy_score, classification_report,
#                              confusion_matrix, f1_score)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_score, recall_score

from config import config

LABELS = config.PARAMS["LABELS"]
SKIP_FRAME_COUNT = config.PARAMS["SKIP_FRAME_COUNT"]
WINDOW_SIZE = config.PARAMS["WINDOW_SIZE"]
num_exercise = config.PARAMS["num_exercise"]
dir_testset = config.PATH["DATASET"]

num_classes_ = config.PARAMS["TOT_CLASSES"]


def load_X(X_path):
    file = open(X_path, "r")
    X = np.array([elem.split(",") for elem in file], dtype=np.float32)
    file.close()
    blocks = int(len(X) / (WINDOW_SIZE * num_exercise))
    X_ = np.array(np.split(X, blocks))
    return X_


# Load the networks outputs
def load_y(y_path):
    file = open(y_path, "r")
    y = np.array(
        [elem for elem in [row.replace("  ", " ").strip().split(" ") for row in file]],
        dtype=np.int32,
    )
    file.close()
    y_ = y - 1
    return y_


# analyse on the images
def test_model_on_testset(X_path, y_path, lstm_classifier, device):

    Xtest = load_X(X_path)
    ytest = load_y(y_path)

    lbs, lbs_gt = [], []
    for i in range(np.array(Xtest).shape[0]):
        # convert input to tensor
        model_input = torch.Tensor(np.array(Xtest[i], dtype=np.float32))
        # add extra dimension
        model_input = torch.unsqueeze(model_input, dim=0)
        if torch.cuda.is_available():
            model_input = model_input.cuda()
        # predict the action class using lstm
        y_pred = lstm_classifier(model_input)
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred_index = prob.data.max(dim=1)[1]
        if not torch.cuda.is_available():
            label = LABELS[pred_index.numpy()[0]]
        label = LABELS[pred_index.detach().cpu().numpy()[0]]
        lbs.append(int(label))
        lb_gt = int(ytest[i][0] + 1)
        lbs_gt.append(lb_gt)

    return lbs, lbs_gt


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_path = dir_testset + "/Xtest.File"
    y_path = dir_testset + "/ytest.File"

    models_name = ["DHAT_LSTM", "tpcnn", "transformer"]

    with open("results/test_results_cls" + str(num_classes_) + ".txt", "w") as f:
        for model_name in models_name:
            model_path = "models/" + model_name + "_scripted_model_3class.pt"

            lstm_classifier = torch.jit.load(model_path).to(device)
            lstm_classifier.eval()

            labels, labels_gt = test_model_on_testset(
                X_path, y_path, lstm_classifier, device
            )

            print("====================================")
            print(f"reuslt of {model_name} model")
            f.write("======================================\n")
            f.write(f"reuslt of {model_name} model \n")
            f.write("======================================\n")
            print("====================================")
            acc = accuracy_score(labels_gt, labels)
            print(f"Test accuracy is: {acc.round(2)}")
            f.write(f"Test accuracy is: {acc.round(2)} \n")

            precision, recall, fscore, support = score(labels_gt, labels)

            print("precision: {} ".format(precision.round(2)))
            f.write("precision: {} \n".format(precision.round(2)))
            print("recall: {}".format(recall.round(2)))
            f.write("recall: {} \n".format(recall.round(2)))
            print("fscore: {}".format(fscore.round(2)))
            f.write("fscore: {} \n".format(fscore.round(2)))
            print("support: {}".format(support))
            f.write("support: {} \n".format(support))

            print(
                "ave f1_score: ", f1_score(labels_gt, labels, average="macro").round(2)
            )
            f.write(
                "ave f1_score: {} \n".format(
                    f1_score(labels_gt, labels, average="macro").round(2)
                )
            )
            print(
                "ave precision: ",
                precision_score(labels_gt, labels, average="macro").round(2),
            )
            f.write(
                "ave precision: {} \n".format(
                    precision_score(labels_gt, labels, average="macro").round(2)
                )
            )
            print(
                "ave recall: {}".format(
                    recall_score(labels_gt, labels, average="macro").round(2)
                )
            )
            f.write(
                "ave recall: {} \n".format(
                    recall_score(labels_gt, labels, average="macro").round(2)
                )
            )
            print("====================================")
