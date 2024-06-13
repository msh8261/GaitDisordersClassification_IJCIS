import math

# import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

WINDOW_SIZE = config.PARAMS["WINDOW_SIZE"]
num_exercise = config.PARAMS["num_exercise"]
num_augment = config.PARAMS["num_augment"]
num_classes_ = config.PARAMS["TOT_CLASSES"]


device = "cuda" if torch.cuda.is_available() else "cpu"


class ATTENTION(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, learning_rate=0.001
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        sequences = WINDOW_SIZE * num_exercise

        self.input_size_h1 = 68
        self.input_size_h2 = input_size - self.input_size_h1

        self.query_h1, self.key_h1, self.value_h1, self.attn_h1, self.sq_d_h1 = (
            self.block(self.input_size_h1)
        )
        self.query_h2, self.key_h2, self.value_h2, self.attn_h2, self.sq_d_h2 = (
            self.block(self.input_size_h2)
        )

        self.drop = nn.Dropout(0.1)
        self.W_0 = nn.Linear(input_size, input_size, bias=False)

    def block(self, input_size_h):
        query_h = nn.Linear(input_size_h, input_size_h, bias=False)
        key_h = nn.Linear(input_size_h, input_size_h, bias=False)
        value_h = nn.Linear(input_size_h, input_size_h, bias=False)
        attn_h = nn.Linear(input_size_h, input_size_h)
        sq_d_h = math.sqrt(input_size_h)
        return query_h, key_h, value_h, attn_h, sq_d_h

    def forward(self, x):

        x1 = x[:, :, : self.input_size_h1]
        x2 = x[:, :, self.input_size_h1 :]

        Q_h1, K_h1, V_h1 = self.query_h1(x1), self.key_h1(x1), self.value_h1(x1)
        # Q.K**T
        energy_h1 = torch.matmul(Q_h1, K_h1.permute(0, 2, 1))
        # Q.K**T/sqrt(d)
        dot_product_h1 = energy_h1 / self.sq_d_h1
        # softmax(Q.K**T/sqrt(d))
        scores_h1 = torch.softmax(dot_product_h1, dim=-1)
        # softmax(Q.K**T/sqrt(d)).V
        scaled_x_h1 = torch.matmul(scores_h1, V_h1) + x1
        out_h1 = self.attn_h1(scaled_x_h1) + x1

        Q_h2, K_h2, V_h2 = self.query_h2(x2), self.key_h2(x2), self.value_h2(x2)
        energy_h2 = torch.matmul(Q_h2, K_h2.permute(0, 2, 1))
        dot_product_h2 = energy_h2 / self.sq_d_h2
        scores_h2 = torch.softmax(dot_product_h2, dim=-1)
        scaled_x_h2 = torch.matmul(scores_h2, V_h2) + x2
        out_h2 = self.attn_h2(scaled_x_h2) + x2

        out = torch.cat([out_h1, out_h2], dim=-1)

        out = self.W_0(out)

        return out


# standard NORM layer of Transformer
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6, trainable=True):
        super(Norm, self).__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        if trainable:
            self.alpha = nn.Parameter(torch.ones(self.size))
            self.bias = nn.Parameter(torch.zeros(self.size))
        else:
            self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        return norm


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=84, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

        nn.init.normal(self.linear_1.weight, std=1)
        nn.init.normal(self.linear_2.weight, std=1)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class TFBlock(nn.Module):
    def __init__(
        self,
        dim,
        hidden_size,
        num_layers,
        num_class,
        heads=2,
        dim_head=None,
        dim_linear_block=84,
        dropout=0.1,
    ):
        super().__init__()
        sequences = WINDOW_SIZE * num_exercise
        dim_linear_block = dim
        self.mhsa = ATTENTION(dim, hidden_size, num_layers, num_class)
        self.drop = nn.Dropout(dropout)

        self.norm_1 = Norm(dim)
        self.norm_2 = Norm(dim)

        self.linear = FeedForward(dim, dim_linear_block)

    def forward(self, x):
        y = self.norm_1(self.drop(self.mhsa(x)) + x)
        return self.norm_2(self.linear(y) + y)


class DHAT_LSTM(pl.LightningModule):
    def __init__(
        self, dim, hidden_size, num_layers, num_class, blocks=2, heads=2, dim_head=None
    ):
        super().__init__()
        self.save_hyperparameters()
        output_size = num_classes_
        sc = 2
        self.block_list = [
            TFBlock(dim, hidden_size, num_layers, num_class, heads, dim_head)
            for _ in range(blocks)
        ]
        self.layers = nn.ModuleList(self.block_list)

        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim * sc,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Linear(dim * sc, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # out, _ = self.lstm(x)
        # out = out[:, -1, :]
        # out = self.fc(out)
        return self.lstm(x)
