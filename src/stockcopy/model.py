import torch.nn as nn

# residual blocks

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            padding = 1,
            bias = False
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            padding = 1,
            bias = False
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        # set residual first
        residual = x

        # convolution layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # combine residual layer
        x = x + residual
        x = self.relu(x)        

        return x


# main nn model

class StockCopy(nn.Module):
    def __init__(self):
        super().__init__()


        self.conv1 = nn.Conv2d(
            in_channels = 104,
            out_channels = 96,
            kernel_size = 3,
            padding = 1,
            bias = False
        )

        self.bn1 = nn.BatchNorm2d(96)

        self.residual_tower = nn.Sequential(
            *[
                ResidualBlock(96) 
                for _ in range(8)
            ]
        )

        self.policy = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 73, kernel_size = 1)
        )

        self.value = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(start_dim = 1),
            nn.Linear(32 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        self.relu = nn.ReLU()

    def forward(self, x):

        # initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # residual tower
        x = self.residual_tower(x)

        # output values
        policy = self.policy(x)
        value = self.value(x)

        return policy, value

