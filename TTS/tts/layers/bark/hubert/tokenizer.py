"""
Custom tokenizer model.
Author: https://www.github.com/gitmylo/
License: MIT
"""

import json
import os.path
from zipfile import ZipFile

import numpy
import torch
from torch import nn, optim


class HubertTokenizer(nn.Module):
    def __init__(self, hidden_size=1024, input_size=768, output_size=10000, version=0):
        super().__init__()
        next_size = input_size
        if version == 0:
            self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
            next_size = hidden_size
        if version == 1:
            self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
            self.intermediate = nn.Linear(hidden_size, 4096)
            next_size = 4096

        self.fc = nn.Linear(next_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.optimizer: optim.Optimizer = None
        self.lossfunc = nn.CrossEntropyLoss()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.version = version

    def forward(self, x):
        x, _ = self.lstm(x)
        if self.version == 1:
            x = self.intermediate(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    @torch.no_grad()
    def get_token(self, x):
        """
        Used to get the token for the first
        :param x: An array with shape (N, input_size) where N is a whole number greater or equal to 1, and input_size is the input size used when creating the model.
        :return: An array with shape (N,) where N is the same as N from the input. Every number in the array is a whole number in range 0...output_size - 1 where output_size is the output size used when creating the model.
        """
        return torch.argmax(self(x), dim=1)

    def prepare_training(self):
        self.optimizer = optim.Adam(self.parameters(), 0.001)

    def train_step(self, x_train, y_train, log_loss=False):
        # y_train = y_train[:-1]
        # y_train = y_train[1:]

        optimizer = self.optimizer
        lossfunc = self.lossfunc
        # Zero the gradients
        self.zero_grad()

        # Forward pass
        y_pred = self(x_train)

        y_train_len = len(y_train)
        y_pred_len = y_pred.shape[0]

        if y_train_len > y_pred_len:
            diff = y_train_len - y_pred_len
            y_train = y_train[diff:]
        elif y_train_len < y_pred_len:
            diff = y_pred_len - y_train_len
            y_pred = y_pred[:-diff, :]

        y_train_hot = torch.zeros(len(y_train), self.output_size)
        y_train_hot[range(len(y_train)), y_train] = 1
        y_train_hot = y_train_hot.to("cuda")

        # Calculate the loss
        loss = lossfunc(y_pred, y_train_hot)

        # Print loss
        if log_loss:
            print("Loss", loss.item())

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

    def save(self, path):
        info_path = ".".join(os.path.basename(path).split(".")[:-1]) + "/.info"
        torch.save(self.state_dict(), path)
        data_from_model = Data(self.input_size, self.hidden_size, self.output_size, self.version)
        with ZipFile(path, "a") as model_zip:
            model_zip.writestr(info_path, data_from_model.save())
            model_zip.close()

    @staticmethod
    def load_from_checkpoint(path, map_location=None):
        old = True
        with ZipFile(path) as model_zip:
            filesMatch = [file for file in model_zip.namelist() if file.endswith("/.info")]
            file = filesMatch[0] if filesMatch else None
            if file:
                old = False
                data_from_model = Data.load(model_zip.read(file).decode("utf-8"))
            model_zip.close()
        if old:
            model = HubertTokenizer()
        else:
            model = HubertTokenizer(
                data_from_model.hidden_size,
                data_from_model.input_size,
                data_from_model.output_size,
                data_from_model.version,
            )
        model.load_state_dict(torch.load(path, map_location=map_location))
        if map_location:
            model = model.to(map_location)
        return model


class Data:
    input_size: int
    hidden_size: int
    output_size: int
    version: int

    def __init__(self, input_size=768, hidden_size=1024, output_size=10000, version=0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.version = version

    @staticmethod
    def load(string):
        data = json.loads(string)
        return Data(data["input_size"], data["hidden_size"], data["output_size"], data["version"])

    def save(self):
        data = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "version": self.version,
        }
        return json.dumps(data)


def auto_train(data_path, save_path="model.pth", load_model: str = None, save_epochs=1):
    data_x, data_y = [], []

    if load_model and os.path.isfile(load_model):
        print("Loading model from", load_model)
        model_training = HubertTokenizer.load_from_checkpoint(load_model, "cuda")
    else:
        print("Creating new model.")
        model_training = HubertTokenizer(version=1).to("cuda")  # Settings for the model to run without lstm
    save_path = os.path.join(data_path, save_path)
    base_save_path = ".".join(save_path.split(".")[:-1])

    sem_string = "_semantic.npy"
    feat_string = "_semantic_features.npy"

    ready = os.path.join(data_path, "ready")
    for input_file in os.listdir(ready):
        full_path = os.path.join(ready, input_file)
        if input_file.endswith(sem_string):
            data_y.append(numpy.load(full_path))
        elif input_file.endswith(feat_string):
            data_x.append(numpy.load(full_path))
    model_training.prepare_training()

    epoch = 1

    while 1:
        for _ in range(save_epochs):
            j = 0
            for x, y in zip(data_x, data_y):
                model_training.train_step(
                    torch.tensor(x).to("cuda"), torch.tensor(y).to("cuda"), j % 50 == 0
                )  # Print loss every 50 steps
                j += 1
        save_p = save_path
        save_p_2 = f"{base_save_path}_epoch_{epoch}.pth"
        model_training.save(save_p)
        model_training.save(save_p_2)
        print(f"Epoch {epoch} completed")
        epoch += 1
