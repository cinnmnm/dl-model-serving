import torch.nn as nn

class CustomMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3),  
            nn.ReLU(),
            nn.Conv2d(32, 16, 3),
            nn.ReLU(),
            nn.Flatten(),   
            nn.Linear(16 * 24 * 24, 64), 
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)