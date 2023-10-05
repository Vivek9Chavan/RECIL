# This is an MLP trained on the embedings received from Pretrained Dino ViT-Small-16

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import sys
import pandas as pd
import numpy as np


class EmbeddingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.embeddings = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        idx = 0

        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.pth'):
                    embedding = torch.load(os.path.join(subdir, file))
                    label = subdir.split('/')[-1]
                    if label not in self.label_to_idx:
                        self.label_to_idx[label] = idx
                        self.idx_to_label[idx] = label
                        idx += 1
                    self.embeddings.append(embedding)
                    self.labels.append(self.label_to_idx[label])

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        if self.transform:
            embedding = self.transform(embedding)
        return embedding, label

train_dataset = EmbeddingDataset(root_dir='/mnt/1TBNVME/MVIP_GrIdL_feats/train/')
val_dataset = EmbeddingDataset(root_dir='/mnt/1TBNVME/MVIP_GrIdL_feats/val/')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)


"""Old Arch
class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(6144, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, input_size)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

"""

class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64*96, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.weight_decay = 1e-5

    def forward(self, x):
        x = x.view(-1, 1, 384)
        out = self.conv1(x)
        out = self.dropout1(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Instantiate the model
input_size = 384
#hidden_size = 256
num_classes = 308
num_epochs = 25
model = CNN(input_size, num_classes)
print(model)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, input_size)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #print every 10 iterations
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'
                    .format(epoch+1, num_epochs, i+1, len(train_dataset)//32, loss.item(), 100*(outputs.argmax(1) == labels).float().mean()))

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.view(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))