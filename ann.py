#%% Importing the libraries

import pandas as pd
import numpy as np

#%% Reading the dataset

df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

#%% Encoding the Gender

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#%% Encoding the country

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#%% Splitting the dataset into training and testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

#%% Scaling the independent variables

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#%% Building the ANN class

import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(12, 6)
        self.fc2 = nn.Linear(6, 6)
        self.output = nn.Linear(6, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

#%% Initializing the model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = ANN().to(device)

#%% Compilation and optimization
import torch.optim as optim

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%% Loading the dataset
from torch.utils.data import DataLoader, TensorDataset

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

batch_size = 32
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#%% Running the model
num_epochs = 200
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print training loss for every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training complete.')

#%% Predicting X_test

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
model.eval()

with torch.no_grad():
    predictions = model(X_test)
    y_pred = (predictions > 0.5).float().cpu()
    
#%% Checking accuracy

from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_np = y_pred.cpu().numpy()
y_test = torch.tensor(y_test, dtype=torch.float32).numpy()
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred_np)








