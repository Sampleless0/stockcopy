import torch
import torch.nn as nn



class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size = 3, padding = 1, bias = False
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size = 3, padding = 1, bias = False
        )
        
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x + residual
        x = self.relu(x)

        return x

    
class ChessResNet(nn.Module):
    def __init__(self,
        input_planes = 20,
        channels = 64,
        num_blocks = 6,
        output_size = 4672
    ):
        super().__init__()


        self.initial = nn.Sequential(
            nn.Conv2d(
                input_planes, channels,
                kernel_size = 3, padding = 1, bias = False
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU()
        )

        self.residual_tower = nn.Sequential(
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels)
        )


        # choosing moves
        self.output = nn.Conv2d(
            channels, 
            73, 
            kernel_size = 1
        )
            

        self.value = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size = 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(start_dim = 1),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.residual_tower(x)

        policy_logits = self.output(x)
        policy_logits = policy_logits.permute(0, 2, 3, 1).contiguous()
        policy_logits = policy_logits.view(x.size(0), -1)

        value = self.value(x)

        return policy_logits, value