import os

#############  Paths and directories   #################################

PATH = {}

# path of raw data included images
PATH["patients"] = "./exported_images_cleaned"
# path of distination files to store the datasets
PATH["dist"] = "./final_data"
# path of labels file
PATH["csv_file"] = "./ORL_skeletons_lookup - labels.csv"

# path of video file
PATH["video_filename"] = "./examples/vid1.avi"

# copy/past the train and test datasets from final_data to data folder to train and test it
PATH["DATASET"] = "./data"

# path to save the models
PATH["lightning_logs"] = "./lightning_logs"


#############  files name   ###########################################

# label file name to store information of patients
label_file_name = "Notice.xml"
# data file name to store features value
data_file_name = "data.File"


#############  Parameters used in codes   #############################

PARAMS = {}
# using different values for different speed of inference vs. performance (inference quality)
# for min_size, the default value is 800
PARAMS["min_size"] = 550  # 800 is default

# number of key points which can be detected by detector
PARAMS["num_keypoints"] = 17
# threshold for person detection
PARAMS["person_thresh"] = 0.7
# threshold for keypoints detection
PARAMS["keypoint_threshold"] = 0
# total number of features in this project
PARAMS["features_size"] = PARAMS["num_keypoints"] * 4 + 8 * 2
# maximum number of frames that would be collected for dataset
PARAMS["MAX_WINDOW_SIZE"] = 100
# this is a limitation to collect features from number of frames defined here
PARAMS["WINDOW_SIZE"] = 100
# We have 3 output action classes.
PARAMS["TOT_CLASSES"] = 3
# there are theree exercise in our dataset
PARAMS["num_exercise"] = 3
# the number of augmentaion which is used in this project
PARAMS["num_augment"] = 5

# how many frames to skip while inferencing
# configuring a higher value will result in better FPS (frames per rate), but accuracy might get impacted
PARAMS["SKIP_FRAME_COUNT"] = 0

# this is used to scaped the features out of frames for our dataset
PARAMS["min_dist_from_frame"] = 10
# here we need to crop image as we need to work on 2d image
PARAMS["image_need_crop"] = True
# scale of wide of frame
PARAMS["scale_w"] = 0.5
# scale of high of frame
PARAMS["scale_h"] = 1

# if need to add zero padding. not useful in this project
PARAMS["ZERO_PADDING"] = False

# these three lines command used for feature engineering
PARAMS["add_speed_angle_features_in_two_sequences"] = True
PARAMS["add_distance_angle_features_in_one_sequence"] = True
PARAMS["add_distance_angle_features_in_two_sequences"] = False

# add augmentation to train dataset
PARAMS["add_augmentation"] = True

# the thickness of points for visualization
PARAMS["points_tickness"] = 3
# color of points
PARAMS["points_COLOR"] = (0, 255, 100)

# device is automatically detected in code
PARAMS["DEVICE"] = "cuda"

# create the list of keypoints.
PARAMS["keypoints"] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


# define of labels or classes
PARAMS["LABELS"] = {0: "1", 1: "2", 2: "3"}
