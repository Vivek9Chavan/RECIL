# Classification using ViT

import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader

import torchvision
import torch.nn as nn
from utlilties import utils, vision_transformer as vits
from PIL import Image

# Load the pre-trained ViT-Small model
model = vits.vit_small(pretrained=True)

# Freeze all layers except the last one

for param in model.parameters():
    param.requires_grad = False

#write classifier
model.classifier = nn.Sequential(
    nn.Linear(384, 10)
)


# Define the number of classes of your dataset
num_classes = 10
#model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Define the data transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the CIFAR dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                           shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                          shuffle=False, num_workers=2)


num_epochs = 10
# Fine-tune the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print every 10 iterations
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'
                    .format(epoch+1, num_epochs, i+1, len(trainset)//128, loss.item(), 100*(outputs.argmax(1) == labels).float().mean()))

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model_{}.ckpt'.format(epoch+1))




