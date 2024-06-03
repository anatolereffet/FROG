from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kernel_size)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size - 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(1, 8, 5)
        self.conv2 = ConvBlock(8, 16, 5)
        self.conv3 = ConvBlock(16, 32, 5)
        self.fc = nn.Linear(32 * 9 * 9, 1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten prior to FC layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SingularTask(nn.Module):
    def __init__(self, finetune: bool = False):
        super().__init__()

        self.finetune = finetune
        self.fc1 = nn.Linear(1024, 128)

        if finetune:
            self.fc2 = nn.Linear(128, 16)
            self.fc3 = nn.Linear(16, 2)
        else:
            self.fc2 = nn.Linear(128, 22)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        if self.finetune:
            x = self.fc3(x)

        return x
