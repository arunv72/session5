import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the neural network
class MNIST_DNN(nn.Module):
    def __init__(self):
        super(MNIST_DNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 19, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(19, 38, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        
        # Third layer (Fully connected)
        self.fc = nn.Linear(38 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 38 * 7 * 7)
        x = self.fc(x)
        return x

# Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load training data
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create the model and define loss function and optimizer
model = MNIST_DNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def count_parameters_by_layer(model):
    params_by_layer = {}
    for name, parameter in model.named_parameters():
        params_by_layer[name] = parameter.numel()
    return params_by_layer

# After model creation, add these lines:
print("\nModel Architecture Details:")
print("Input size: 28 x 28 (MNIST standard)")

# Calculate and display parameters by layer
params_by_layer = count_parameters_by_layer(model)
print("\nParameters by layer:")
for name, count in params_by_layer.items():
    print(f"{name}: {count:,} parameters")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Calculate theoretical memory usage
memory_bits = total_params * 32  # 32 bits per parameter
memory_mb = memory_bits / (8 * 1024 * 1024)  # Convert to MB
print(f"Approximate model size: {memory_mb:.2f} MB")

print("\nLayer shapes:")
print(f"Conv1: 1 -> 19 channels, 3x3 kernel")
print(f"Conv2: 19 -> 38 channels, 3x3 kernel")
print(f"FC: {38 * 7 * 7} -> 10 neurons")

print("\nStarting training...")

# Training loop
def train(model, train_loader, criterion, optimizer):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            current_accuracy = 100 * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            print(f'Batch [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Running Loss: {avg_loss:.4f}, '
                  f'Current Accuracy: {current_accuracy:.2f}%')
    
    final_accuracy = 100 * correct / total
    print(f'Final Training Accuracy: {final_accuracy:.2f}%')
    return final_accuracy

# Train for one epoch
accuracy = train(model, train_loader, criterion, optimizer) 