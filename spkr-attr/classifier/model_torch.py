import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from clearml import Task
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class baseline_model(nn.Module):
    def __init__(self, class_num, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, class_num)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        output = self.dropout1(F.relu(self.fc1(x)))
        output = self.dropout2(F.relu(self.fc2(output)))
        output = F.relu(self.fc3(output))
        output = F.relu(self.fc4(output))
        output = F.softmax(self.fc5(output), dim=0)
        return output


def load_data(train, labels):
    t = torch.Tensor(np.load(train))
    l = torch.Tensor(np.load(labels))
    dataset = torch.utils.data.TensorDataset(t, l)
    return dataset


def compute_metrics(y_pred, y_true):
    acc = accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    print(f"Model Accuracy on untrained data: {acc}")
    conf_matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    print(f"Confusion matrix on {y_true.shape[-1]} classes: \n {conf_matrix}")
    f1 = f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average="weighted")
    print(f"Weighted f1 score: {f1}")


def test_model(test, test_labels, model):
    test, y_true = load_data(test, test_labels)
    y_pred = model.predict(test, verbose=1)
    compute_metrics(y_pred, y_true)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--mode", help="train or evaluate", required=True)
    parser.add_argument("--embedding_dim", default=512)
    parser.add_argument(
        "--load_ckpt",
    )
    parser.add_argument(
        "--class_weights",
    )
    parser.add_argument("--epoch", default=20)
    parser.add_argument("--batch", default=50)
    parser.add_argument(
        "--clearml_project",
        default="YourTTS-sprint2",
    )
    parser.add_argument(
        "--clearml_task",
        default="attribute-classifier",
    )

    args = parser.parse_args()
    Task.init(project_name=args.clearml_project, task_name=f"{args.clearml_task}-{args.labels}")
    # Load data
    dataset = load_data(args.data, args.labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2)

    # Initialize model
    class_num = len(np.unique(np.load(args.labels), axis=0))
    model = baseline_model(class_num, args.embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train
    for epoch in range(args.epoch):

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            print(f"[Epoch:{epoch + 1}, step:{i + 1:5d}] loss: {loss.item() :.3f}")

    checkpoint_dir = "checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_filepath = f"{checkpoint_dir}/checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_filepath)


if __name__ == "__main__":
    main()
