# import glob
import os
import random
import sys
# import shutil
import xml.etree.ElementTree as ET

# import src.augmentation as aug
# import src.feature_selection as fs
from config import config as config

# import cv2
# import numpy as np


# from xml.dom import minidom
# from xml.etree import ElementTree
# from xml.etree.ElementTree import Comment, Element, SubElement, tostring
# from nn.keypointrcnn_resnet50_fpn import keypointrcnn_resnet50_fpn


sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

# configure the parameters
dir_patients = config.PATH["dist"]


def merge_and_save_files(files, dst_data_path):
    # Reading data from file1
    with open(dst_data_path, "w") as fw:
        for file in files:
            with open(file, "r") as fr:
                data = fr.read()
                fw.write(data)
                # fw.write("\n")


def save_lable_file(xml_file, dst_label_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    y = root.find("evaluation").attrib["evaluation"]
    with open(dst_label_path, "w") as fy:
        fy.write(y)


def main():

    try:
        base_folders = [
            name
            for name in os.listdir(dir_patients)
            if os.path.isdir(os.path.join(dir_patients, name))
        ]
        assert len(base_folders) != 0
    except:
        print("INFO: Please check if the directory of source exist.")

    for folder_patient_id in base_folders:
        base_dir = os.path.join(dir_patients, folder_patient_id)
        folders_patient_date_train = [
            name
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        ]
        num_subfolders = len(folders_patient_date_train)
        if num_subfolders > 1:
            folder_patient_date_val = random.choice(folders_patient_date_train)
            folders_patient_date_train.remove(folder_patient_date_val)
            base_dir = os.path.join(
                dir_patients, folder_patient_id, folder_patient_date_val
            )
            Label_name = (
                folder_patient_id + "_" + folder_patient_date_val + "_label.File"
            )
            data_name = (
                folder_patient_id + "_" + folder_patient_date_val + "_data_val.File"
            )
            dst_label_path = os.path.join(dir_patients, folder_patient_id, Label_name)
            dst_data_path = os.path.join(dir_patients, folder_patient_id, data_name)
            data_files_path = [
                os.path.join(base_dir, name)
                for name in os.listdir(base_dir)
                if name.endswith(".File")
            ]
            merge_and_save_files(data_files_path, dst_data_path)
            xml_file = os.path.join(base_dir, "Notice.xml")
            save_lable_file(xml_file, dst_label_path)

            print(f"INFO: Data val for {folder_patient_date_val} is written...")

        for folder_date in folders_patient_date_train:
            base_dir = os.path.join(dir_patients, folder_patient_id, folder_date)
            Label_name = folder_patient_id + "_" + folder_date + "_label.File"
            data_name = folder_patient_id + "_" + folder_date + "_data_train.File"
            dst_label_path = os.path.join(dir_patients, folder_patient_id, Label_name)
            dst_data_path = os.path.join(dir_patients, folder_patient_id, data_name)
            data_files_path = [
                os.path.join(base_dir, name)
                for name in os.listdir(base_dir)
                if name.endswith(".File")
            ]
            merge_and_save_files(data_files_path, dst_data_path)
            xml_file = os.path.join(base_dir, "Notice.xml")
            save_lable_file(xml_file, dst_label_path)

            print(f"INFO: Data train for {folder_date} is written...")


if __name__ == "__main__":
    main()
