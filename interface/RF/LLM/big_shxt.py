import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
start_time = time.time()

# ==========================================
# 1. Environment Check & Setup
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Success! GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Warning: No GPU detected. Falling back to CPU.")

# ==========================================
# 2. Define Classes (Dataset & Model)
# ==========================================
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    def __len__(self):
        return self.labels.shape[0]

class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs): 
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, num_outputs),
        )
    def forward(self, x):
        return self.layers(x)

# ==========================================
# 3. Data Preparation & Model Setup
# ==========================================
# Prepare dummy data
X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]])
y_train = torch.tensor([0, 0, 0, 1, 1])

# Create DataLoader
train_loader = DataLoader(
    dataset=ToyDataset(X_train, y_train),
    batch_size=2,
    shuffle=True
)

# Initialize Model and move to GPU
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# ==========================================
# 4. Execute Training Loop (GPU Test)
# ==========================================
print("\n Starting GPU training loop test...")

for epoch in range(3):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        # Key step: Move data to the selected device (GPU)
        features, labels = features.to(device), labels.to(device)
        
        # Standard training steps
        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"   Epoch {epoch+1}/3 Complete | Loss: {loss.item():.4f}")
print("\n Test finished! If no errors occurred and Loss is displayed, your GPU setup is perfect.")

end = time.time()
print(f'{end - start_time} seconds')