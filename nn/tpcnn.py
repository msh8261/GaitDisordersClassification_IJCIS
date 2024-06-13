import pytorch_lightning as pl
import torch
import torch.nn as nn

from config import config

WINDOW_SIZE = config.PARAMS["WINDOW_SIZE"]
num_exercise = config.PARAMS["num_exercise"]


device = "cuda" if torch.cuda.is_available() else "cpu"


class TPCNN(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, learning_rate=0.001
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        sequences = WINDOW_SIZE * num_exercise
        in_c = sequences

        input_size1 = 68
        input_size2 = 8
        input_size3 = 8

        self.conv_layer1 = self._conv_layer_set(in_c, input_size1, 3)
        self.conv_layer2 = self._conv_layer_set(input_size1, 64, 3)
        self.conv_layer3 = self._conv_layer_set(64, 128, 2)
        self.conv_layer4 = self._conv_layer_set(128, 256, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 3, 128)
        self.drop = nn.Dropout(0.25)

        self.conv_layer1_2 = self._conv_layer_set(in_c, input_size2, 2)
        self.conv_layer2_2 = self._conv_layer_set(input_size2, 64, 2)

        self.conv_layer1_3 = self._conv_layer_set(in_c, input_size3, 2)
        self.conv_layer2_3 = self._conv_layer_set(input_size3, 64, 2)

        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(64 * 1, 128)

        # 128 * number of block
        self.fc3 = nn.Linear(128 * 3, output_size)
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _conv_layer_set(in_c, out_c, kernel_size):
        conv_layer = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
        )
        return conv_layer

    def block1(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        # print("=============Block1==============")
        # print(out.shape)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.drop(out)
        return out

    def block2(self, x):
        out = self.conv_layer1_2(x)
        out = self.conv_layer2_2(out)
        # print("============Block2===============")
        # print(out.shape)
        out = self.flatten(out)
        # print(out.shape)
        out = self.fc2(out)
        # print(out.shape)
        out = self.drop(out)
        return out

    def block3(self, x):
        out = self.conv_layer1_3(x)
        out = self.conv_layer2_3(out)
        # print("============Block3===============")
        # print(out.shape)
        out = self.flatten(out)
        out = self.fc2(out)
        out = self.drop(out)
        return out

    def forward(self, x):
        x1 = x[:, :, :68]
        x2 = x[:, :, 68:76]
        x3 = x[:, :, 76:]
        out1 = self.block1(x1)
        out2 = self.block2(x2)
        out3 = self.block2(x3)
        # print("============Block1,2,3===============")
        # print(out1.shape)
        # print(out2.shape)
        # print(out3.shape)
        # to train multi models
        out = torch.cat((out1, out2, out3), dim=1)
        print(out.shape)
        out = self.softmax(self.fc3(out))

        return out
