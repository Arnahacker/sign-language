"""
Smoke test — runs in CI without a trained model.
Creates a dummy HybridModel, exports it to TorchScript,
then tests both the C++ engine and Python inference path.
"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import HybridModel

NUM_CLASSES  = 5
DUMMY_FRAMES = [[float(i % 10) / 10.0] * 258 for i in range(30)]

# ── 1. Export a dummy TorchScript model ────────────────────────────────────────
print("Creating dummy HybridModel...")
model = HybridModel(input_size=258, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES)
model.eval()

dummy_input = torch.zeros(1, 30, 258)
with torch.no_grad():
    scripted = torch.jit.trace(model, dummy_input)

os.makedirs("Final_model_parameters", exist_ok=True)
scripted.save("Final_model_parameters/action_scripted.pt")
print("TorchScript export: OK")

# ── 2. C++ engine test (only if built) ────────────────────────────────────────
try:
    import sign_inference as si
    engine = si.SignInferenceEngine("Final_model_parameters/action_scripted.pt")
    result = engine.predict(DUMMY_FRAMES)

    assert 0 <= result.index < NUM_CLASSES, f"Index out of range: {result.index}"
    assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range: {result.confidence}"

    print(f"C++ inference: OK  (index={result.index}, confidence={result.confidence:.4f})")
except ImportError:
    print("C++ module not found — skipping C++ test (build first with build.sh)")

# ── 3. Python TorchScript inference test ──────────────────────────────────────
loaded = torch.jit.load("Final_model_parameters/action_scripted.pt")
loaded.eval()

with torch.no_grad():
    out   = loaded(dummy_input)
    probs = torch.softmax(out, dim=1)

assert probs.shape == (1, NUM_CLASSES), f"Unexpected output shape: {probs.shape}"
assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilities don't sum to 1"

print("Python TorchScript inference: OK")
print("\nAll smoke tests passed.")
