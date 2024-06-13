import ast
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

import src.augmentation as aug
# from xml.etree.ElementTree import Comment, Element, SubElement, tostring
from config import config

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

WINDOW_SIZE = config.PARAMS["WINDOW_SIZE"]

dir_dist = config.PATH["dist"]


def prepare_list_to_save_in_file(p_arr_dist_angle_added):
    p_str = [[str(a) for a in b] for b in p_arr_dist_angle_added]
    p_list = [item for sublist in p_str for item in sublist]
    p_list_to_file = ["%s" % item for item in p_list]
    p_list_to_file = ",".join(p_list_to_file)
    return p_list_to_file


def data_augmentation(data):
    data = np.array(data).reshape(1, np.array(data).shape[0], np.array(data).shape[1])
    x1 = aug.jitter(data)
    x2 = aug.scaling(data)
    x3 = aug.magnitude_warp(data)
    x4 = aug.time_warp(data)
    x5 = aug.rotation(data)
    x6 = aug.window_slice(data)

    XX = np.vstack((data, x1, x2, x3, x4, x5))
    px_str = [[[(c) for c in a] for a in b] for b in XX]
    px = [item for sublist in px_str for item in sublist]
    px = [[str(item) for item in line] for line in px]
    px = [",".join(line) for line in px]

    return px


def label_augmentation(target):
    y = np.array(target)
    y = y.reshape(y.shape[0], 1)
    yy = np.vstack((y, y, y, y, y, y))
    py_str = [[str(a) for a in b] for b in yy]

    py = [item for sublist in py_str for item in sublist]
    py = ["%s" % item for item in py]
    py = "".join(py)

    return py


def data_augmentation_and_save_to_file(base_dir, folder_patient_id, folder_date):
    X_in_file_name = folder_patient_id + "_" + folder_date + "_data_train.File"
    path_X_in = os.path.join(base_dir, X_in_file_name)
    X_out_file_name = (
        folder_patient_id + "_" + folder_date + "_data_train_augmentated.File"
    )
    path_X_out = os.path.join(base_dir, X_out_file_name)
    y_in_file_name = folder_patient_id + "_" + folder_date + "_label.File"
    path_y_in = os.path.join(base_dir, y_in_file_name)
    y_out_file_name = folder_patient_id + "_" + folder_date + "_label_augmentated.File"
    path_y_out = os.path.join(base_dir, y_out_file_name)

    with open(path_X_in, "r") as fx:
        Xtrain = []
        for line in fx:
            # print(line.split())
            if len(line.split()) > 0:
                str_line = line.split()[0]
                # print(str_line)
                list_line = list(ast.literal_eval(str_line))
                Xtrain.append(list_line)
            else:
                print("INFO: Error of empty data!")

        px = data_augmentation(Xtrain)

        with open(path_X_out, "w") as fx:
            for line in px:
                fx.write(line)
                fx.write("\n")

    with open(path_y_in, "r") as fy:
        ytrain = []
        for line in fy:
            str_line = line.split()[0]
            ytrain.append(str_line)

        py = label_augmentation(ytrain)

        with open(path_y_out, "w") as fy:
            for line in py:
                fy.write(line)
                fy.write("\n")


def merge_and_save_files(files, dst_data_path):
    with open(dst_data_path, "w") as fw:
        for file in files:
            with open(file, "r") as fr:
                data = fr.read()
                fw.write(data)
                # fw.write("\n")


def main():

    base_folders = [
        name
        for name in os.listdir(dir_dist)
        if os.path.isdir(os.path.join(dir_dist, name))
    ]

    for folder_patient_id in base_folders:
        base_dir1 = os.path.join(dir_dist, folder_patient_id)
        folders_patient_date = [
            name
            for name in os.listdir(base_dir1)
            if os.path.isdir(os.path.join(base_dir1, name))
        ]
        for folder_date in folders_patient_date:
            if os.path.isfile(
                base_dir1
                + "/"
                + folder_patient_id
                + "_"
                + folder_date
                + "_data_val.File"
            ):
                print(folder_patient_id + "_" + folder_date + "_data_val.File")
                continue
            print(os.path.join(folder_patient_id, folder_date))
            data_augmentation_and_save_to_file(
                base_dir1, folder_patient_id, folder_date
            )


if __name__ == "__main__":
    main()
