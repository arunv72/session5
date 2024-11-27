import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist_dnn import MNIST_DNN
import json
from datetime import datetime

def validate_model():
    # Initialize model and training components
    model = MNIST_DNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Training loop
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    
    # Validation results
    results = {
        "accuracy": accuracy,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Save results
    with open('validation_results.json', 'w') as f:
        json.dump(results, f)
    
    # Save model with timestamp and accuracy
    model_name = f"model_{results['timestamp']}_acc{accuracy:.2f}.pth"
    torch.save(model.state_dict(), model_name)
    
    return accuracy >= 95.0

if __name__ == "__main__":
    success = validate_model()
    exit(0 if success else 1) 