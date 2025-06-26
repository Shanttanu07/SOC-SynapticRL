import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_inputs = 28 * 28
num_hidden = 100
num_outputs = 2
beta = 0.9
batch_size = 64
num_epochs = 5
learning_rate = 1e-3

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Filter dataset to only include digits 0 and 1
def filter_digits(dataset, digits=[0, 1]):
    idx = (dataset.targets == digits[0]) | (dataset.targets == digits[1])
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    dataset.targets = (dataset.targets == digits[1]).long()
    return dataset

# Load datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset = filter_digits(train_dataset)
test_dataset = filter_digits(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the SNN model
class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, surrogate_function=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, surrogate_function=surrogate.fast_sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        mem1 = self.lif1.init_leaky()
        spk1, mem1 = self.lif1(self.fc1(x), mem1)
        mem2 = self.lif2.init_leaky()
        spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
        return spk2

# Instantiate model, loss, optimizer
model = SNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
print("\nEvaluating...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
