import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split


DATA_PATH = os.path.join('MP_Data_10') 
actions = np.array(os.listdir(DATA_PATH))
label_map = {label:num for num, label in enumerate(actions)}

print(f"Detected Actions: {actions}")

sequences, labels = [], []

print("Loading data with Augmentation")

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.isdir(action_path): continue
        
    seq_folders = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
    
    for sequence in seq_folders:
        window = []
        try:
            for frame_num in range(30):
                res = np.load(os.path.join(action_path, sequence, "{}.npy".format(frame_num)))
                window.append(res)
            
            original_sequence = np.array(window) # (30, 258)
            sequences.append(original_sequence)
            labels.append(label_map[action])
            noise = np.random.normal(0, 0.02, original_sequence.shape) 
            sequences.append(original_sequence + noise)
            labels.append(label_map[action])
            scale = np.random.uniform(0.95, 1.05) # +/- 5% size
            sequences.append(original_sequence * scale)
            labels.append(label_map[action])

        except Exception as e:
            pass

print(f"Total Sequences: {len(sequences)}")

X = np.array(sequences)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# Convert to Torch
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(HybridModel, self).__init__()
        
        self.feature_extract = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(30), 
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.feature_extract(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context_vector)
        return out


INPUT_SIZE = 258 
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = len(actions)

model = HybridModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# --- 3. TRAINING LOOP ---
EPOCHS = 80 # 150 is plenty with the augmented data
print(f"Starting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_acc = 100 * correct / total
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | Acc: {epoch_acc:.2f}%')

# --- 4. TEST & SAVE ---
print("\n🧪 Testing Model...")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Final Test Accuracy: {100 * correct / total:.2f}%")

# ✅ FIX 4: SAFE RELATIVE PATHS (No more "Directory Not Found")
current_folder = os.getcwd()
save_dir = os.path.join(current_folder, 'Final_model_parameters')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, 'action.pth')
labels_path = os.path.join(save_dir, 'labels.npy')

torch.save(model.state_dict(), model_path)
np.save(labels_path, actions)

print(f"Model saved to: {model_path}")
print("IMPORTANT: Copy this 'HybridModel' class to your api.py before running the server!")