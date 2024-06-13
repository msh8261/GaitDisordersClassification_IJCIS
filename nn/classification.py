import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from config import config
from nn.dhat import DHAT_LSTM
from nn.tpcnn import TPCNN
from nn.transformer import TRANSFORMER

WINDOW_SIZE = config.PARAMS["WINDOW_SIZE"]
num_exercise = config.PARAMS["num_exercise"]
num_augment = config.PARAMS["num_augment"]
num_classes_ = config.PARAMS["TOT_CLASSES"]


device = "cuda" if torch.cuda.is_available() else "cpu"

# weights for different class
weights = torch.tensor([1.0, 2.0, 2.0]).to(device)


factor_ = 0.5
patience_ = 50
min_lr_ = 1e-15


class Classification(pl.LightningModule):
    """classification models"""

    def __init__(
        self,
        model_name,
        input_size,
        hidden_size,
        num_layers,
        num_class,
        learning_rate=0.001,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.lr = learning_rate

        self.model_name = model_name

        if model_name == "tpcnn":
            model = TPCNN(input_size, hidden_size, num_layers, num_class)
        elif model_name == "DHAT_LSTM":
            model = DHAT_LSTM(input_size, hidden_size, num_layers, num_class)
        elif model_name == "transformer":
            heads = 2
            model = TRANSFORMER(input_size, hidden_size, heads, num_class)
        else:
            assert print("INFO: model name is not defined")

        self.model = model

    def forward(self, x):
        model = self.model
        out = model.forward(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        # reduce dimension
        y = torch.squeeze(y)
        # convert to long
        y = y.long()
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y, weight=weights)
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        acc = torchmetrics.functional.accuracy(pred, y)
        f1_score = torchmetrics.functional.f1(
            pred, y, num_classes=num_classes_, average="weighted"
        )

        dic = {
            "batch_train_loss": loss,
            "batch_train_acc": acc,
            "batch_train_f1": f1_score,
        }
        self.log("batch_train_loss", loss, prog_bar=True)
        self.log("batch_train_acc", acc, prog_bar=True)
        self.log("batch_train_f1", f1_score, prog_bar=True)
        return {"loss": loss, "result": dic}

    def training_epoch_end(self, training_step_outputs):
        # calculate average training loss end of the epoch
        avg_train_loss = torch.tensor(
            [x["result"]["batch_train_loss"] for x in training_step_outputs]
        ).mean()
        # calculate average training accuracy end of the epoch
        avg_train_acc = torch.tensor(
            [x["result"]["batch_train_acc"] for x in training_step_outputs]
        ).mean()
        avg_train_f1 = torch.tensor(
            [x["result"]["batch_train_f1"] for x in training_step_outputs]
        ).mean()
        self.log("train_loss", avg_train_loss, prog_bar=True)
        self.log("train_acc", avg_train_acc, prog_bar=True)
        self.log("train_f1", avg_train_f1, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y)
        y = y.long()
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y, weight=weights)

        prob = F.softmax(y_pred, dim=1)
        pred = prob.data.max(dim=1)[1]
        acc = torchmetrics.functional.accuracy(pred, y)
        f1_score = torchmetrics.functional.f1(
            pred, y, num_classes=num_classes_, average="weighted"
        )

        dic = {
            "batch_val_loss": loss,
            "batch_val_acc": acc,
            "batch_val_f1": f1_score,
        }
        self.log("batch_val_loss", loss, prog_bar=True, logger=True)
        self.log("batch_val_acc", acc, prog_bar=True, logger=True)
        self.log("batch_val_f1", f1_score, prog_bar=True, logger=True)
        return dic

    def validation_epoch_end(self, validation_step_outputs):
        avg_val_loss = torch.tensor(
            [x["batch_val_loss"] for x in validation_step_outputs]
        ).mean()
        avg_val_acc = torch.tensor(
            [x["batch_val_acc"] for x in validation_step_outputs]
        ).mean()
        avg_val_f1 = torch.tensor(
            [x["batch_val_f1"] for x in validation_step_outputs]
        ).mean()
        self.log("val_loss", avg_val_loss, prog_bar=True)
        self.log("val_acc", avg_val_acc, prog_bar=True)
        self.log("val_f1", avg_val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y)
        y = y.long()
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)

        prob = F.softmax(y_pred, dim=1)
        pred = prob.data.max(dim=1)[1]
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {"batch_test_loss": loss, "batch_test_acc": acc}
        self.log("batch_test_loss", loss, prog_bar=True)
        self.log("batch_test_acc", acc, prog_bar=True)
        return dic

    def test_epoch_end(self, test_step_outputs):
        avg_test_loss = torch.tensor(
            [x["batch_test_loss"] for x in test_step_outputs]
        ).mean()
        avg_test_acc = torch.tensor(
            [x["batch_test_acc"] for x in test_step_outputs]
        ).mean()
        self.log("test_loss", avg_test_loss, prog_bar=True)
        self.log("test_acc", avg_test_acc, prog_bar=True)

    def configure_optimizers(self):
        # adam optimiser
        # add l2 regularization to optimzer by just adding in a weight_decay
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor_,
            patience=patience_,
            min_lr=min_lr_,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }
