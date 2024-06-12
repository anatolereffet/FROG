from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=ch_out)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size - 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.maxpool(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(3, 8, 5)
        self.conv2 = ConvBlock(8, 16, 5)
        self.conv3 = ConvBlock(16, 32, 5)
        # Adapt the feature map of the paper to our img input
        self.fc = nn.Linear(32 * 24 * 24, 1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten prior to FC layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extract = ConvNet()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.hard_dropout = nn.Dropout(0.5)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.feat_extract(x)
        x = self.hard_dropout(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x
