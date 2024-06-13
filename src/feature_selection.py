# import math
# import os

from math import atan2, degrees

# import numpy as np
from scipy.spatial import distance
from sklearn import preprocessing
from torchvision import transforms as T

# from stat import FILE_ATTRIBUTE_REPARSE_POINT
from config import config as config

min_dist_from_frame = config.PARAMS["min_dist_from_frame"]


def image_scaling(frame, image_need_crop, scale_w, scale_h):
    # make a copy of the frame
    img = frame.copy()
    h, w = img.shape[:2]
    if image_need_crop:
        w = int(w * scale_w)
        h = int(h * scale_h)
        img = img[0:h, 0:w]
    return h, w, img


def input_for_model(img, device):
    # preprocess the input image
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)
    # add one dim with unsqueeze
    return img_tensor.unsqueeze_(0)


def filter_persons(model_output, person_thresh):
    persons = {}
    p_indicies = [i for i, s in enumerate(model_output["scores"]) if s > person_thresh]
    for i in p_indicies:
        desired_kp = model_output["keypoints"][i][:].to("cpu")
        persons[i] = desired_kp
    return (persons, p_indicies)


def check_to_get_all_features_available_in_image(h, w, keypoints):
    arr = keypoints[0].detach().cpu().numpy()
    arr = [[a for a in b] for b in arr]
    res = [
        [
            (
                1
                if (
                    min_dist_from_frame < arr[i][0] < (w - min_dist_from_frame)
                    and min_dist_from_frame < arr[i][1] < (h - min_dist_from_frame)
                )
                else 0
            )
        ]
        for i in range(len(arr))
    ]
    res = [item for sublist in res for item in sublist]
    flag = all(res)
    return flag, arr


def normalize_values_from_image(arr, h, w):
    arr = [[float(arr[i][0] / w), float(arr[i][1] / h)] for i in range(len(arr))]
    return arr


def Angle_Btw_2Points(pointA, pointB):
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    return (degrees(atan2(changeInY, changeInX)) + 180) / 360


def add_distance_angle_of_keypoints_in_two_sequences(arr_last, arr_current):
    arr_dist = [
        distance.euclidean(arr_last[i], arr_current[i]) for i in range(len(arr_last))
    ]
    arr_angle = [
        Angle_Btw_2Points(arr_last[i], arr_current[i]) for i in range(len(arr_last))
    ]
    final_arr = [
        [arr_current[i][0], arr_current[i][1], arr_dist[i], arr_angle[i]]
        for i in range(len(arr_current))
    ]
    final_arr = [item for sublist in final_arr for item in sublist]
    return final_arr


def add_speed_angle_of_keypoints_in_two_sequences(dtime, arr_last, arr_current):
    if dtime <= 0:
        dtime = 1
    arr_speed = [
        (distance.euclidean(arr_last[i], arr_current[i]) / dtime)
        for i in range(len(arr_last))
    ]
    arr_speed = preprocessing.normalize([arr_speed], norm="l2")[0]
    arr_angle = [
        Angle_Btw_2Points(arr_last[i], arr_current[i]) for i in range(len(arr_last))
    ]
    final_arr = [
        [arr_current[i][0], arr_current[i][1], arr_speed[i], arr_angle[i]]
        for i in range(len(arr_current))
    ]
    final_arr = [item for sublist in final_arr for item in sublist]
    return final_arr


def add_distance_angle_of_symetric_keypoints_in_a_sequence(arr, keypoints):
    left_eye = keypoints[1]
    right_eye = keypoints[2]

    left_ear = keypoints[3]
    right_ear = keypoints[4]

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    left_elbow = keypoints[7]
    right_elbow = keypoints[8]

    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    left_hip = keypoints[11]
    right_hip = keypoints[12]

    left_knee = keypoints[13]
    right_knee = keypoints[14]

    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    left_side = [
        left_eye,
        left_ear,
        left_shoulder,
        left_elbow,
        left_wrist,
        left_hip,
        left_knee,
        left_ankle,
    ]
    right_side = [
        right_eye,
        right_ear,
        right_shoulder,
        right_elbow,
        right_wrist,
        right_hip,
        right_knee,
        right_ankle,
    ]

    features = [[left_side[i], right_side[i]] for i in range(len(right_side))]
    arr_dist = [
        distance.euclidean(left_side[i], right_side[i]) for i in range(len(left_side))
    ]
    arr_angle = [
        Angle_Btw_2Points(left_side[i], right_side[i]) for i in range(len(left_side))
    ]
    features = [[arr_dist[i], arr_angle[i]] for i in range(len(left_side))]

    features = [item for sublist in features for item in sublist]
    arr.extend(features)
    return arr
