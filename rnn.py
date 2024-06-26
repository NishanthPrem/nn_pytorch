#%% Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

#%% Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%% Load and preprocess the training data
df_train = pd.read_csv('Google_Stock_Price_Train.csv')
train_set = df_train.iloc[:, 1:2].values

#%% Feature Scaling using Min-Max Normalization
sc = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = sc.fit_transform(train_set)

#%% Create sequences of 60 timesteps
X_train, y_train = [], []
for i in range(60, len(train_set_scaled)):
    X_train.append(train_set_scaled[i-60:i, 0])
    y_train.append(train_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the data to be compatible with PyTorch's expectations
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#%% Define the PyTorch LSTM model
class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(RNNRegressor, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        
        self.lstm4 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout4 = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        out, _ = self.lstm3(out)
        out = self.dropout3(out)
        
        out, _ = self.lstm4(out)
        out = self.dropout4(out)
        
        out = self.fc(out[:, -1, :])  # We are only interested in the last output
        return out

# Define model parameters
input_size = 1
hidden_size = 50
num_layers = 1
dropout = 0.2

# Instantiate the model and move it to the GPU if available
model = RNNRegressor(input_size, hidden_size, num_layers, dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%% Prepare data for PyTorch DataLoader
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#%% Training loop
num_epochs = 100
model.train()  # Set the model to training mode

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Clear the gradients
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")

#%% Load and preprocess the test data
df_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_price = df_test.iloc[:, 1:2].values

df_total = pd.concat((df_train['Open'], df_test['Open']), axis=0)
inputs = df_total[len(df_total) - len(df_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# Prepare the test data
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Convert test data to tensor and move to GPU
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

#%% Make predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predicted_price = model(X_test_tensor)
    predicted_price = predicted_price.cpu().numpy()  # Move predictions back to CPU

# Inverse transform the predicted values to the original scale
predicted_price = sc.inverse_transform(predicted_price)

# Plot the results
plt.plot(real_price, color='red', label='Actual Google Stock Price')
plt.plot(predicted_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
