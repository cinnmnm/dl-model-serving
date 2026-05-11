import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import bentoml

# 1. Define CNN for MNIST
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 2. Setup Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3. Setup Training
model = MNIST_CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

target_accuracy = 98.7
current_accuracy = 0.0
epoch = 1

print(f"Training until model reaches {target_accuracy}% accuracy...")

# 4. Training Loop
while current_accuracy < target_accuracy:
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Evaluate Accuracy
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    current_accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Epoch {epoch} complete. Current Accuracy: {current_accuracy:.2f}%")
    epoch += 1

# 5. Save the winning model to BentoML
print("Target reached! Saving model to BentoML...")
bentoml.pytorch.save_model("mnist_model", model)
print("Model saved successfully.")