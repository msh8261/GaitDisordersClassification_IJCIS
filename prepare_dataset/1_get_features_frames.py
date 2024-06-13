import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
from xml.etree import ElementTree

import cv2
import numpy as np
import torch

import src.feature_selection as fs
import src.image_filters as fl
import src.kalman as kalman
# from xml.etree.ElementTree import Comment, Element, SubElement, tostring
from config import config as config
from nn.keypointrcnn_resnet50_fpn import keypointrcnn_resnet50_fpn

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

# configure the parameters
dir_patients = config.PATH["patients"]
if not dir_patients.endswith("/"):
    dir_patients += "/"
dir_dist = config.PATH["dist"]
path_csv_file = config.PATH["csv_file"]

# WINDOW_SIZE = config.PARAMS["WINDOW_SIZE"]
WINDOW_SIZE = config.PARAMS["MAX_WINDOW_SIZE"]
keypoints = config.PARAMS["keypoints"]

image_need_crop = config.PARAMS["image_need_crop"]
scale_w = config.PARAMS["scale_w"]
scale_h = config.PARAMS["scale_h"]
ZERO_PADDING = config.PARAMS["ZERO_PADDING"]
person_thresh = config.PARAMS["person_thresh"]
keypoint_threshold = config.PARAMS["keypoint_threshold"]
num_keypoints = config.PARAMS["num_keypoints"]

add_speed_angle_features_in_two_sequences = config.PARAMS[
    "add_speed_angle_features_in_two_sequences"
]
add_distance_angle_features_in_two_sequences = config.PARAMS[
    "add_distance_angle_features_in_two_sequences"
]
add_distance_angle_features_in_one_sequence = config.PARAMS[
    "add_distance_angle_features_in_one_sequence"
]

bins_dir = "C:/Users/mohsen/Desktop/Postdoc_Upa/Datasets/GaitAnalysis/raw data"


def find_time_bin(bin_file_name):
    data = bin_file_name.split("/")
    end = data[-1].replace(".bin", "")
    mins = end.split("_")
    timeSec = (
        int(mins[4].split("\\")[1]) * 60 * 60
        + int(mins[5]) * 60
        + int(mins[6])
        + int(mins[7]) / 1000.0
    )
    return timeSec


def find_time(img_file_name):
    data = img_file_name.split("/")[-1:][0].split("\\")[-1:][0]
    end = data.replace(".jpg", "")
    mins = end.split("_")
    timeSec = (
        int(mins[0]) * 3600 + int(mins[1]) * 60 + int(mins[2]) + int(mins[3]) / 1000.0
    )
    return timeSec


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ElementTree.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def prepare_xml_for_labels(params, path_to_save="items.xml"):
    # create the file structure
    data = ET.Element("measurementInfo")
    data.set("beforeSurgery", params["beforeSurgery"])
    data.set("idPatient", params["idPatient"])
    data.set("valid", params["valid"])

    Date_ = ET.SubElement(data, "measurementDate")
    Date_.set("Date", params["Date"])

    evaluation_ = ET.SubElement(data, "evaluation")
    evaluation_.set("evaluation", params["evaluation"])

    # create a new XML file with the results
    myfile = open(path_to_save, "w")
    mydata = prettify(data)
    myfile.write(mydata)


def get_labels_ids_from_csv(path_csv_file):
    with open(path_csv_file, "r") as file:
        list_of_ids = []
        list_of_labels = []
        lines = file.readlines()[1:]
        for line in lines:  # read rest of lines
            line = [int(x) for x in line.split(",")]
            list_of_ids.append(line[0])
            list_of_labels.append(line[1])

    return list_of_ids, list_of_labels


# prepare the list of features to be saved in the text file
def prepare_list_to_save_in_file(p_arr_dist_angle_added):
    p_list = [str(item) for item in p_arr_dist_angle_added]
    p_list_to_file = ["%s" % item for item in p_list]
    p_list_to_file = ",".join(p_list_to_file)
    return p_list_to_file


def write_lists_to_files(list1, list2, path1, path2):
    with open(path1, "w") as f:
        for item in list1:
            f.write("%s\n" % item)
        print("Done")

    with open(path2, "w") as f:
        for item in list2:
            f.write("%s\n" % item)
        print("Done")


def add_zero_padding(features_size):
    padd = np.zeros(features_size)
    padd = [str(val) for val in padd]
    padd = ",".join(padd)
    return padd


def select_best_filter_result(model, filtered_image_mix, filtered_image_hist, device):

    h, w, img_mix = fs.image_scaling(
        filtered_image_mix, image_need_crop, scale_w, scale_h
    )
    h, w, img_hist = fs.image_scaling(
        filtered_image_hist, image_need_crop, scale_w, scale_h
    )

    img_tensor_mix = fs.input_for_model(img_mix, device)
    img_tensor_hist = fs.input_for_model(img_hist, device)

    output_mix = model(img_tensor_mix)[0]
    output_hist = model(img_tensor_hist)[0]
    persons_mix, p_inds_mix = fs.filter_persons(output_mix, person_thresh)
    persons_hist, p_inds_hist = fs.filter_persons(output_hist, person_thresh)

    keypoints_scores_mix = output_mix["keypoints_scores"].cpu().detach().numpy()
    keypoints_scores_hist = output_hist["keypoints_scores"].cpu().detach().numpy()
    if (len(persons_hist) == 1) and len(keypoints_scores_hist) > 0:
        if len(keypoints_scores_hist[p_inds_hist][0]) == config.PARAMS["num_keypoints"]:
            return persons_hist, p_inds_hist, keypoints_scores_hist, img_hist
    elif (len(persons_mix) == 1) and len(keypoints_scores_mix) > 0:
        return persons_mix, p_inds_mix, keypoints_scores_mix, img_mix
    else:
        return persons_hist, p_inds_hist, keypoints_scores_hist, img_hist


def point2xyv(kp):
    kp = np.array(kp)
    x = kp[0::3].astype(int)
    y = kp[1::3].astype(int)
    v = kp[2::3].astype(int)  # visibility, 0 = Not visible, 0 != visible
    return x, y, v


def initialize_kalman(num_keypoints):
    """initialize kalman filter for 17 keypoints"""
    list_KFs = []
    for i in range(num_keypoints):
        KF = kalman.KF2d(dt=1)  # time interval: '1 frame'
        init_P = 1 * np.eye(4, dtype=np.float64)  # Error cov matrix
        init_x = np.array(
            [0, 0, 0, 0], dtype=np.float64
        )  # [x loc, x vel, y loc, y vel]
        dict_KF = {"KF": KF, "P": init_P, "x": init_x}
        list_KFs.append(dict_KF)
    return list_KFs, KF


def points_tracking(list_KFs, KF, keypoints):
    list_estimate = []  # kf filtered keypoints
    keypoints = keypoints[0].detach().cpu().numpy()
    for i in range(len(keypoints)):
        # print(keypoints[i])
        kx = keypoints[i][0]
        ky = keypoints[i][1]
        z = np.array([kx, ky], dtype=np.float64)

        KF = list_KFs[i]["KF"]
        x = list_KFs[i]["x"]
        P = list_KFs[i]["P"]

        x, P, filtered_point = KF.process(x, P, z)

        list_KFs[i]["KF"] = KF
        list_KFs[i]["x"] = x
        list_KFs[i]["P"] = P

        # visibility
        v = 0 if filtered_point[0] == 0 and filtered_point[1] == 0 else 2
        list_estimate.extend(list(filtered_point) + [v])  # x,y,v

    return list_estimate


# apply keypoints detector to get features from a patient moving front of camera
def prepare_data_file_from_images_bins(
    model, output_data_train, final_folder_images, final_folder_bins, dist, device
):
    folder_name = os.path.split(os.path.split(dist)[0])[0]
    print(folder_name)
    with open(output_data_train, "w") as fx:
        count = 0
        ct = 0
        last_time = 0
        norm_arr_last = np.zeros((len(keypoints), 2))

        for i, img_path in enumerate(final_folder_images):
            bin_file = final_folder_bins[i]
            real_time_frame = find_time_bin(bin_file)

            count += 1
            filename = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            filtered_image_mix = fl.apply_hist_colormap_filter(frame)
            filtered_image_hist = fl.apply_equalhist_filter(frame)

            persons, p_inds, keypoints_scores, img = select_best_filter_result(
                model, filtered_image_mix, filtered_image_hist, device
            )
            h, w = img.shape[:2]

            current_time = real_time_frame

            if len(persons) == 1 and len(keypoints_scores) > 0 and ct < WINDOW_SIZE:
                # add kalman filter to tracking the points
                list_KFs, KF = initialize_kalman(num_keypoints)
                list_estimate = points_tracking(list_KFs, KF, persons)
                list_estimate = [
                    [
                        list_estimate[i * 3],
                        list_estimate[(i * 3) + 1],
                        list_estimate[(i * 3) + 2],
                    ]
                    for i in range(len(list_estimate))
                    if i < len(list_estimate) / 3
                ]
                persons = torch.tensor([list_estimate]).to(device)
                flag, points_arr = fs.check_to_get_all_features_available_in_image(
                    h, w, persons
                )
                if bool(flag):
                    dtime = (current_time - last_time) + 1e-6
                    print("Times: ", current_time, ", ", last_time)
                    last_time = current_time
                    norm_arr_current = fs.normalize_values_from_image(points_arr, h, w)
                    if (
                        add_speed_angle_features_in_two_sequences
                        and add_distance_angle_features_in_one_sequence
                    ):
                        p_arr = fs.add_speed_angle_of_keypoints_in_two_sequences(
                            dtime, norm_arr_last, norm_arr_current
                        )
                        p_arr = (
                            fs.add_distance_angle_of_symetric_keypoints_in_a_sequence(
                                p_arr, norm_arr_current
                            )
                        )
                        p_list_to_file = prepare_list_to_save_in_file(p_arr)
                    elif (
                        add_distance_angle_features_in_two_sequences
                        and add_distance_angle_features_in_one_sequence
                    ):
                        p_arr = fs.add_distance_angle_of_keypoints_in_two_sequences(
                            norm_arr_last, norm_arr_current
                        )
                        p_arr = (
                            fs.add_distance_angle_of_symetric_keypoints_in_a_sequence(
                                p_arr, norm_arr_current
                            )
                        )
                        p_list_to_file = prepare_list_to_save_in_file(p_arr)
                    elif add_distance_angle_features_in_two_sequences:
                        p_arr = fs.add_distance_angle_of_keypoints_in_two_sequences(
                            norm_arr_last, norm_arr_current
                        )
                        p_list_to_file = prepare_list_to_save_in_file(p_arr)
                    elif add_distance_angle_features_in_one_sequence:
                        p_arr = [
                            item for sublist in norm_arr_current for item in sublist
                        ]
                        p_arr = (
                            fs.add_distance_angle_of_symetric_keypoints_in_a_sequence(
                                p_arr, norm_arr_current
                            )
                        )
                        p_list_to_file = prepare_list_to_save_in_file(p_arr)
                    else:
                        p_arr = [
                            item for sublist in norm_arr_current for item in sublist
                        ]
                        p_list_to_file = prepare_list_to_save_in_file(p_arr)

                    norm_arr_last = norm_arr_current
                    ct += 1
                    fx.writelines(p_list_to_file)
                    fx.write("\n")
                    cv2.imwrite(os.path.join(dist, filename), img)

        print("before: ", count, ct)
        collect = 0
        if 0 < ct < WINDOW_SIZE:
            collect = ct
            for i in range(WINDOW_SIZE - ct):
                ct += 1
                if ZERO_PADDING:
                    features_size = config.PARAMS["features_size"]
                    padding = add_zero_padding(features_size)
                    fx.writelines(padding)
                else:
                    fx.writelines(p_list_to_file)
                fx.write("\n")

            print("after: ", count, ct)
            return collect
        elif ct == 0:
            print("after: ", count, ct)
            return 0


# apply keypoints detector to get features from a patient moving front of camera
def prepare_data_file_from_images(
    model, output_data_train, final_folder_images, dist, device
):
    folder_name = os.path.split(os.path.split(dist)[0])[0]
    print(folder_name)
    with open(output_data_train, "w") as fx:
        count = 0
        ct = 0
        last_time = 0
        first_detection = True
        norm_arr_last = np.zeros((len(keypoints), 2))

        for i, img_path in enumerate(final_folder_images):

            real_time_frame = find_time(img_path)
            print("time of recorded image: ", real_time_frame)

            count += 1
            filename = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            filtered_image_mix = fl.apply_hist_colormap_filter(frame)
            filtered_image_hist = fl.apply_equalhist_filter(frame)

            persons, p_inds, keypoints_scores, img = select_best_filter_result(
                model, filtered_image_mix, filtered_image_hist, device
            )
            h, w = img.shape[:2]

            current_time = real_time_frame

            if len(persons) == 1 and len(keypoints_scores) > 0 and ct < WINDOW_SIZE:
                # add kalman filter to tracking the points
                list_KFs, KF = initialize_kalman(num_keypoints)
                list_estimate = points_tracking(list_KFs, KF, persons)
                list_estimate = [
                    [
                        list_estimate[i * 3],
                        list_estimate[(i * 3) + 1],
                        list_estimate[(i * 3) + 2],
                    ]
                    for i in range(len(list_estimate))
                    if i < len(list_estimate) / 3
                ]
                persons = torch.tensor([list_estimate]).to(device)
                flag, points_arr = fs.check_to_get_all_features_available_in_image(
                    h, w, persons
                )
                if bool(flag):
                    dtime = (current_time - last_time) + 0.0000001
                    if first_detection:
                        dtime = 0

                    first_detection = False

                    last_time = current_time

                    norm_arr_current = fs.normalize_values_from_image(points_arr, h, w)
                    if (
                        add_speed_angle_features_in_two_sequences
                        and add_distance_angle_features_in_one_sequence
                    ):
                        p_arr = fs.add_speed_angle_of_keypoints_in_two_sequences(
                            dtime, norm_arr_last, norm_arr_current
                        )
                        p_arr = (
                            fs.add_distance_angle_of_symetric_keypoints_in_a_sequence(
                                p_arr, norm_arr_current
                            )
                        )
                        p_list_to_file = prepare_list_to_save_in_file(p_arr)
                    elif (
                        add_distance_angle_features_in_two_sequences
                        and add_distance_angle_features_in_one_sequence
                    ):
                        p_arr = fs.add_distance_angle_of_keypoints_in_two_sequences(
                            norm_arr_last, norm_arr_current
                        )
                        p_arr = (
                            fs.add_distance_angle_of_symetric_keypoints_in_a_sequence(
                                p_arr, norm_arr_current
                            )
                        )
                        p_list_to_file = prepare_list_to_save_in_file(p_arr)
                    elif add_distance_angle_features_in_two_sequences:
                        p_arr = fs.add_distance_angle_of_keypoints_in_two_sequences(
                            norm_arr_last, norm_arr_current
                        )
                        p_list_to_file = prepare_list_to_save_in_file(p_arr)
                    elif add_distance_angle_features_in_one_sequence:
                        p_arr = [
                            item for sublist in norm_arr_current for item in sublist
                        ]
                        p_arr = (
                            fs.add_distance_angle_of_symetric_keypoints_in_a_sequence(
                                p_arr, norm_arr_current
                            )
                        )
                        p_list_to_file = prepare_list_to_save_in_file(p_arr)
                    else:
                        p_arr = [
                            item for sublist in norm_arr_current for item in sublist
                        ]
                        p_list_to_file = prepare_list_to_save_in_file(p_arr)

                    norm_arr_last = norm_arr_current
                    ct += 1
                    fx.writelines(p_list_to_file)
                    fx.write("\n")
                    cv2.imwrite(os.path.join(dist, filename), img)

        print("before: ", count, ct)
        collect = 0
        if 0 < ct < WINDOW_SIZE:
            collect = ct
            for i in range(WINDOW_SIZE - ct):
                ct += 1
                if ZERO_PADDING:
                    features_size = config.PARAMS["features_size"]
                    padding = add_zero_padding(features_size)
                    fx.writelines(padding)
                else:
                    fx.writelines(p_list_to_file)
                fx.write("\n")

            print("after: ", count, ct)
            return collect
        elif ct == 0:
            print("after: ", count, ct)
            return 0


def process_data_and_save_to_file(
    base_folders, ids, labels, model, device, save_bad_folders_to_file
):
    params = {}
    list_low_detection = []
    list_failed_folders = []
    for folder_patient_id in base_folders:
        id = int(folder_patient_id.lstrip("0"))
        print("id: ", id)
        if id in ids:
            ind = ids.index(id)
            label = labels[ind]
            base_dir = os.path.join(dir_patients, folder_patient_id)
            folders_patient_date = [
                name
                for name in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, name))
            ]
            os.makedirs(os.path.join(dir_dist, folder_patient_id), exist_ok=True)
            dist_1 = os.path.join(dir_dist, folder_patient_id)
            for folder_date in folders_patient_date:
                base_dir = os.path.join(dir_patients, folder_patient_id, folder_date)
                folders_patient_exercise = [
                    name
                    for name in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, name))
                ]
                os.makedirs(os.path.join(dist_1, folder_date), exist_ok=True)
                dist_2 = os.path.join(dist_1, folder_date)
                params["beforeSurgery"] = "True"
                params["idPatient"] = str(folder_patient_id)
                params["valid"] = "True"
                params["Date"] = str(folder_date)
                params["evaluation"] = str(label)
                for folder_exercise in folders_patient_exercise:
                    base_dir = os.path.join(
                        dir_patients, folder_patient_id, folder_date, folder_exercise
                    )
                    final_folder = [
                        name
                        for name in os.listdir(base_dir)
                        if os.path.isdir(os.path.join(base_dir, name))
                    ][0]
                    os.makedirs(
                        os.path.join(dist_2, folder_exercise, final_folder),
                        exist_ok=True,
                    )
                    path_xml_to_save = os.path.join(dist_2, config.label_file_name)
                    prepare_xml_for_labels(params, path_xml_to_save)
                    dist_save_images = os.path.join(
                        dist_2, folder_exercise, final_folder
                    )
                    final_folder_images = [
                        os.path.join(base_dir, final_folder, name)
                        for name in os.listdir(os.path.join(base_dir, final_folder))
                    ]
                    # base_dir_bins = os.path.join(bins_dir, folder_patient_id, folder_date, folder_exercise)
                    # final_folder_bins = [os.path.join(base_dir_bins, name) for name in os.listdir(base_dir_bins)]
                    data_file_name = (
                        folder_patient_id
                        + "_"
                        + folder_date
                        + "_"
                        + folder_exercise
                        + "_"
                        + config.data_file_name
                    )
                    # output_data_train = os.path.join(dist_2, folder_exercise, data_file_name)
                    output_data_train = os.path.join(dist_2, data_file_name)
                    # flag = prepare_data_file_from_images_bins(model, output_data_train, final_folder_images, final_folder_bins, dist_save_images, device)
                    flag = prepare_data_file_from_images(
                        model,
                        output_data_train,
                        final_folder_images,
                        dist_save_images,
                        device,
                    )

                    if flag == 0:
                        info2 = os.path.join(
                            folder_patient_id, folder_date, folder_exercise
                        )
                        info = info2 + ":" + str(flag)
                        print("list_failed_folders: ", info)
                        list_failed_folders.append(info)
                    elif flag != None and flag != 0:
                        info1 = os.path.join(
                            folder_patient_id, folder_date, folder_exercise
                        )
                        info = info1 + ":" + str(flag)
                        print("list_low_detection: ", info)
                        list_low_detection.append(info)
                    print(f"Data for {folder_exercise} is written...")

    if save_bad_folders_to_file:
        path1 = "./list_low_detection_final.txt"
        path2 = "./list_failed_folders_final.txt"
        write_lists_to_files(list_low_detection, list_failed_folders, path1, path2)


def main():
    print(dir_patients)

    try:
        base_folders = [
            name
            for name in os.listdir(dir_patients)
            if os.path.isdir(os.path.join(dir_patients, name))
        ]
    except:
        print("INFO: Please check if the directory of source exist.")

    model, device = keypointrcnn_resnet50_fpn.create()

    ids, labels = get_labels_ids_from_csv(path_csv_file)

    save_bad_folders_to_file = True

    start = time.perf_counter()
    process_data_and_save_to_file(
        base_folders, ids, labels, model, device, save_bad_folders_to_file
    )
    end = time.perf_counter()
    print(f"Finished in {round(end-start, 2)} seconds")


if __name__ == "__main__":
    main()
