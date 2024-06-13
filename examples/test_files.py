import argparse
# import string
import sys

# import opencv liberary
import cv2
# import torch liberary
import torch
import torchvision
from torchvision import transforms as T

# import local file
from src.draw_skeleton import draw_skeleton_per_person


def skeleton_detection_video(vid_path):
    # create opencv class
    cap = cv2.VideoCapture(vid_path)
    if cap.isOpened() == False:
        print("INFO: Error opening video stream")

    # create a model object from the keypointrcnn_resnet50_fpn class
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    if torch.cuda.is_available():
        print("INFO: python running with cuda....")
    else:
        print("INFO: python running with cpu....")

    while True:
        # read the frame
        ret, img0 = cap.read()
        if ret == False:
            break

        img = img0.copy()

        # preprocess the input image
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img).to(device)
        # apply model on frame
        output = model([img_tensor])[0]
        # call the local function to show the result
        skeletal_img = draw_skeleton_per_person(
            img,
            output["scores"],
            output["keypoints"],
            output["keypoints_scores"],
            keypoint_threshold=2,
            conf_threshold=0.9,
        )

        cv2.imshow("win", skeletal_img)
        if cv2.waitKey(1) == 27:
            break

    # release the video capture object
    cap.release()
    cv2.destroyAllWindows()


def skeleton_detection_image(img_path):

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    if torch.cuda.is_available():
        print("INFO: python running with cuda....")
    else:
        print("INFO: python running with cpu....")

    # read the frame
    img0 = cv2.imread(img_path)
    if img0 is None:
        print("INFO: Error opening image file")

    img = img0.copy()

    # preprocess the input image
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)
    # apply model on frame
    output = model([img_tensor])[0]
    # call the local function to show the result
    skeletal_img = draw_skeleton_per_person(
        img,
        output["scores"],
        output["keypoints"],
        output["keypoints_scores"],
        keypoint_threshold=2,
        conf_threshold=0.9,
    )

    cv2.imshow("image", skeletal_img)
    cv2.waitKey(0)
    # release the video capture object
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # path of the video for test
    file_path = sys.argv[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", help="path to input iamge")
    ap.add_argument("--video_path", help="path to input video")
    args = ap.parse_args()

    if args.video_path is not None:
        # function to call pretrained model to show the person skeleton model on video
        skeleton_detection_video(args.video_path)
    elif args.image_path is not None:
        # function to call pretrained model to show the person skeleton model on image
        skeleton_detection_image(args.image_path)
