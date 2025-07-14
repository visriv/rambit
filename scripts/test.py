import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ===== Setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dummy data: 100 samples, 10 timesteps, 5 features
X = torch.randn(100, 10, 5)
y = torch.randint(0, 2, (100,))

# Wrap in DataLoader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ===== GRU Model =====
class SimpleGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=5, hidden_size=8, batch_first=True)
        self.fc = nn.Linear(8, 2)  # Binary classification

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))

model = SimpleGRU().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ===== Training Loop =====
for epoch in range(3):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
