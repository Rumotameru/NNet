import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3)
        self.norm1 = nn.BatchNorm2d(12)
        self.conv1_drop = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3)
        self.norm2 = nn.BatchNorm2d(24)
        self.conv2_drop = nn.Dropout2d(0.2)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=3)
        self.norm3 = nn.BatchNorm2d(48)
        self.conv3_drop = nn.Dropout2d(0.2)
        self.dense_drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(17*17*48, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):

        x = F.relu(self.conv1_drop(F.max_pool2d(self.norm1(self.conv1(x)), 2)))
        x = F.relu(self.conv2_drop(F.max_pool2d(self.norm2(self.conv2(x)), 2)))
        x = F.relu(self.conv3_drop(F.max_pool2d(self.norm3(self.conv3(x)), 2)))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense_drop(self.fc1(x)))
        x = F.elu(self.dense_drop(self.fc2(x)))
        x = self.fc3(x)
        return x


def train(model, loader, t_loss, opt, criterion, device):
    for data, target in loader:
        # move-tensors-to-GPU
        data = data.to(device)
        target = target.to(device)
        # target = target.unsqueeze(1)
        # target = target.float()
        # clear-the-gradients-of-all-optimized-variables
        opt.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        opt.step()
        # update-training-loss
        t_loss += loss.item() * data.size(0)

    return model, t_loss, opt


def valid(model, loader, v_loss, criterion, device):
    correct = 0
    total = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        # target = target.unsqueeze(1)
        # target = target.float()
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        loss = criterion(output, target)

        # update-average-validation-loss
        v_loss += loss.item() * data.size(0)
    acc = accuracy('Valid',correct, total)
    return model, v_loss, acc


def test(model, loader, device):
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().float()
    acc = accuracy('Test', correct, total)
    return acc


def accuracy(process, correct, total):
    print('\t{} Accuracy of the model: {} %'.format(process, 100 * correct/total))
    return correct/total

