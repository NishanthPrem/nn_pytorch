#%%
import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

#%% Loading the training set

train_transforms = v2.Compose([
    v2.Resize((64,64)),
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=0,shear=0.2,scale=(0.8, 1.2)),
    v2.ToTensor(),
    v2.Normalize((0.5,), (0.5,))])

train_set = datasets.ImageFolder(
    root='dataset/training_set',
    transform=train_transforms)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=5)

#%% Loading the test set

test_transforms = v2.Compose([
    v2.Resize((64,64)),
    v2.ToTensor(),
    v2.Normalize((0.5,), (0.5,))])

test_set = datasets.ImageFolder(
    root='dataset/test_set',
    transform=test_transforms)

test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=5)

#%% Creating the CNN model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=32 * 16 * 16, out_features=128)  # Adjust based on input size after pooling
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))   # Apply first conv layer and ReLU activation
        x = self.pool(x)               # Apply first pooling layer
        x = self.relu(self.conv2(x))   # Apply second conv layer and ReLU activation
        x = self.pool(x)               # Apply second pooling layer
        x = x.view(-1, 32 * 16 * 16)   # Flatten the tensor for the fully connected layer
        x = self.relu(self.fc1(x))     # Apply first fully connected layer and ReLU activation
        x = self.sigmoid(self.fc2(x))  # Apply second fully connected layer and Sigmoid activation
        return x

#%% Initializing the CNN model 
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%% Training and testing the CNN model 

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(
            device).float().view(-1, 1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

print("Training complete!")
    
  
    
    
    
    
    
    
    
    
    