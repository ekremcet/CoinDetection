import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

TRAIN_DIR = './Coins/TrainData'
TEST_DIR = './Coins/TestData'
IMG_SIZE = 256


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), 16 * 61 * 61)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def load_data(data_dir):
    data_transforms = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor(), ])
    data_folder = datasets.ImageFolder(data_dir, transform=data_transforms)

    num_train = len(data_folder)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)
    data = torch.utils.data.DataLoader(data_folder, sampler=sampler, batch_size=64)

    return data


train_data = load_data(TRAIN_DIR)
# Initialize Model and Optimizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_data, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 4 == 3:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 3))
            running_loss = 0.0


torch.save(model, 'coinmodel.pth')
print('Finished Training')

test_data = load_data(TEST_DIR)
# Test Model
correct = 0
total = 0
with torch.no_grad():
    for data in test_data:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))