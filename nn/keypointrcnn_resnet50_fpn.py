import torch
import torchvision

from config import config as config


class keypointrcnn_resnet50_fpn:
    @staticmethod
    def create():
        # create a model object from the keypointrcnn_resnet50_fpn class
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            pretrained=True,
            num_keypoints=config.PARAMS["num_keypoints"],
            min_size=config.PARAMS["min_size"],
        )
        # set the computation device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load the model on to the computation device and set to eval mode
        model.to(device).eval()

        return model, device
