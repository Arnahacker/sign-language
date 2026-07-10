import torch
import numpy as np
import os
from model import HybridModel

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Final_model_parameters", "action.pth")
LABELS_PATH = os.path.join(BASE_DIR, "Final_model_parameters", "labels.npy")
EXPORT_PATH = os.path.join(BASE_DIR, "Final_model_parameters", "action_scripted.pt")

# ── Load labels ────────────────────────────────────────────────────────────────
actions    = np.load(LABELS_PATH)
num_classes = len(actions)
print(f"Classes: {num_classes} → {actions}")

# ── Rebuild model and load weights ─────────────────────────────────────────────
device = torch.device("cpu")   # export on CPU; LibTorch will handle device at runtime

model = HybridModel(
    input_size=258,
    hidden_size=64,
    num_layers=2,
    num_classes=num_classes
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("Weights loaded.")

# ── Export to TorchScript via tracing ─────────────────────────────────────────
# trace needs a representative input: batch=1, frames=30, features=258
dummy_input = torch.zeros(1, 30, 258)

with torch.no_grad():
    scripted = torch.jit.trace(model, dummy_input)

# ── Verify outputs match ───────────────────────────────────────────────────────
test_input = torch.randn(1, 30, 258)
with torch.no_grad():
    original_out = model(test_input)
    scripted_out = scripted(test_input)

max_diff = (original_out - scripted_out).abs().max().item()
print(f"Max output difference (original vs scripted): {max_diff:.2e}")
assert max_diff < 1e-5, "Outputs don't match — check the model forward pass"
print("Outputs match.")

# ── Save ──────────────────────────────────────────────────────────────────────
scripted.save(EXPORT_PATH)
print(f"TorchScript model saved to: {EXPORT_PATH}")
