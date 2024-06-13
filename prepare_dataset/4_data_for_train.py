import os
import sys

from config import config

# import xml.etree.ElementTree as ET
# from xml.etree.ElementTree import Comment, Element, SubElement, tostring


sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

WINDOW_SIZE = config.PARAMS["WINDOW_SIZE"]

dir_dist = config.PATH["dist"]


def merge_and_save_files(files, dst_data_path):
    with open(dst_data_path, "w") as fw:
        for file in files:
            with open(file, "r") as fr:
                data = fr.read()
                fw.write(data)


def merge_and_save_labels(files, dst_label_path):
    with open(dst_label_path, "w") as fw:
        for file in files:
            with open(file, "r") as fr:
                data = fr.read()
                fw.write(data)
                fw.write("\n")


def main():

    base_folders = [
        name
        for name in os.listdir(dir_dist)
        if os.path.isdir(os.path.join(dir_dist, name))
    ]

    list_data_augmentated = []
    list_label_augmentated = []
    list_data_val = []
    list_label_val = []
    for folder_patient_id in base_folders:
        base_dir1 = os.path.join(dir_dist, folder_patient_id)
        folders_patient_date = [
            name
            for name in os.listdir(base_dir1)
            if os.path.isdir(os.path.join(base_dir1, name))
        ]
        # data_Files_path = [os.path.join(base_dir1, name) for name in os.listdir(base_dir1) if name.endswith('.File')]
        for folder_date in folders_patient_date:
            data_val_path = (
                base_dir1
                + "/"
                + folder_patient_id
                + "_"
                + folder_date
                + "_data_val.File"
            )
            data_label_path = (
                base_dir1 + "/" + folder_patient_id + "_" + folder_date + "_label.File"
            )
            if os.path.isfile(data_val_path):
                if os.path.isfile(data_label_path):
                    list_data_val.append(data_val_path)
                    list_label_val.append(data_label_path)
                    continue

            data_augmentated_path = (
                base_dir1
                + "/"
                + folder_patient_id
                + "_"
                + folder_date
                + "_data_train_augmentated.File"
            )
            label_augmentated_path = (
                base_dir1
                + "/"
                + folder_patient_id
                + "_"
                + folder_date
                + "_label_augmentated.File"
            )
            if os.path.isfile(data_augmentated_path):
                if os.path.isfile(label_augmentated_path):
                    list_data_augmentated.append(data_augmentated_path)
                    list_label_augmentated.append(label_augmentated_path)

    dst_data_train_path = os.path.join(dir_dist, "Xtrain.File")
    dst_label_train_path = os.path.join(dir_dist, "ytrain.File")
    merge_and_save_files(list_data_augmentated, dst_data_train_path)
    merge_and_save_labels(list_label_augmentated, dst_label_train_path)

    dst_data_val_path = os.path.join(dir_dist, "Xtest.File")
    dst_label_val_path = os.path.join(dir_dist, "ytest.File")
    merge_and_save_files(list_data_val, dst_data_val_path)
    merge_and_save_labels(list_label_val, dst_label_val_path)


if __name__ == "__main__":
    main()
