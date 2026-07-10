import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from model import HybridModel

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join("MP_Data_10")
EPOCHS     = 80
BATCH_SIZE = 32

actions   = np.array(os.listdir(DATA_PATH))
label_map = {label: num for num, label in enumerate(actions)}
print(f"Detected actions: {actions}")

# ── Load data with augmentation ────────────────────────────────────────────────
sequences, labels = [], []
print("Loading data with augmentation...")

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.isdir(action_path):
        continue

    seq_folders = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]

    for sequence in seq_folders:
        window = []
        try:
            for frame_num in range(30):
                res = np.load(os.path.join(action_path, sequence, f"{frame_num}.npy"))
                window.append(res)

            original = np.array(window)

            # Original
            sequences.append(original)
            labels.append(label_map[action])

            # Noise augmentation
            sequences.append(original + np.random.normal(0, 0.02, original.shape))
            labels.append(label_map[action])

            # Scale augmentation
            sequences.append(original * np.random.uniform(0.95, 1.05))
            labels.append(label_map[action])

        except Exception as e:
            print(f"Skipping {action}/{sequence}: {e}")

print(f"Total sequences: {len(sequences)}")

# ── Prepare tensors ────────────────────────────────────────────────────────────
X = np.array(sequences)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test  = torch.FloatTensor(X_test)
y_test  = torch.LongTensor(y_test)

train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(torch.utils.data.TensorDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

# ── Model ──────────────────────────────────────────────────────────────────────
model     = HybridModel(input_size=258, hidden_size=64, num_layers=2, num_classes=len(actions))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# ── Training loop ──────────────────────────────────────────────────────────────
print(f"Training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = torch.max(outputs.data, 1)
        total        += batch_labels.size(0)
        correct      += (predicted == batch_labels).sum().item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | Acc: {100*correct/total:.2f}%")

# ── Evaluation ─────────────────────────────────────────────────────────────────
print("\nTesting model...")
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for inputs, batch_labels in test_loader:
        outputs      = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total       += batch_labels.size(0)
        correct     += (predicted == batch_labels).sum().item()

print(f"Test accuracy: {100 * correct / total:.2f}%")

# ── Save weights ───────────────────────────────────────────────────────────────
save_dir = os.path.join(os.getcwd(), "Final_model_parameters")
os.makedirs(save_dir, exist_ok=True)

model_path  = os.path.join(save_dir, "action.pth")
labels_path = os.path.join(save_dir, "labels.npy")

torch.save(model.state_dict(), model_path)
np.save(labels_path, actions)
print(f"Weights saved to: {model_path}")

# ── Export TorchScript ─────────────────────────────────────────────────────────
model.eval()
dummy_input = torch.zeros(1, 30, 258)

with torch.no_grad():
    scripted = torch.jit.trace(model, dummy_input)

scripted_path = os.path.join(save_dir, "action_scripted.pt")
scripted.save(scripted_path)
print(f"TorchScript model saved to: {scripted_path}")
