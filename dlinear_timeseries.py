import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
import wandb  # Import Weights & Biases

# Initialize Weights & Biases
wandb.init(project="qlabintern2025", name="dlinear_timeseries", config={})

# Load saved train and test data
train_data = np.load("train_data.npy")
test_data = np.load("test_data.npy")

print(f"Train Data Shape: {train_data.shape}")
print(f"Test Data Shape: {test_data.shape}")

# Hyperparameters
sequence_length = 96
prediction_length = 14
epochs = 50
batch_size = 32
learning_rate = 0.001

# Log hyperparameters in W&B
wandb.config.update({
    "sequence_length": sequence_length,
    "prediction_length": prediction_length,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate
})

# Prepare data
def create_sequences(data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length : i + seq_length + pred_length, -1])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, sequence_length, prediction_length)
X_test, y_test = create_sequences(test_data, sequence_length, prediction_length)

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Define DLinear Model
class DLinear(nn.Module):
    def __init__(self, input_size):
        super(DLinear, self).__init__()
        self.linear = nn.Linear(input_size * sequence_length, prediction_length)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.linear(x)

# Initialize Model
model = DLinear(input_size=X_train.shape[2])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()

    # Log loss to W&B
    wandb.log({"epoch": epoch + 1, "loss": loss.item()})

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Evaluate Model
with torch.no_grad():
    y_pred = model(X_test).numpy()
    y_true = y_test.numpy()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

# Log final metrics to W&B
wandb.log({"MSE": mse, "MAE": mae})

print(f"\nFinal Results:\nMSE: {mse:.6f}\nMAE: {mae:.6f}")

# Save model
model_path = "dlinear_model_v2.pth"
torch.save(model.state_dict(), model_path)
print("\nModel saved successfully as 'dlinear_model_v2.pth'")

# Upload model to W&B as an artifact
artifact = wandb.Artifact("dlinear_model_v2", type="model")
artifact.add_file(model_path)
wandb.log_artifact(artifact)

print("\nModel uploaded to Weights & Biases as an artifact.")
