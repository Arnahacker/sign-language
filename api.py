from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os
from typing import List
from model import HybridModel
from converting_words_to_sentences import OllamaHandler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
SCRIPTED_MODEL_PATH  = os.path.join(BASE_DIR, "Final_model_parameters", "action_scripted.pt")
MODEL_PATH           = os.path.join(BASE_DIR, "Final_model_parameters", "action.pth")
LABELS_PATH          = os.path.join(BASE_DIR, "Final_model_parameters", "labels.npy")

CONFIDENCE_THRESHOLD = 0.85
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model      = None
actions    = None
llm        = None
cpp_engine = None  # C++ inference engine if available

# ── Try loading the C++ engine ─────────────────────────────────────────────────
try:
    import sign_inference as _si
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    print("C++ module not found — run build.sh to enable it. Falling back to Python.")


def load_resources():
    global model, actions, llm, cpp_engine

    # Load labels
    if os.path.exists(LABELS_PATH):
        actions = np.load(LABELS_PATH)
        print(f"Loaded {len(actions)} labels")
    else:
        print(f"Labels not found at {LABELS_PATH}")
        actions = np.array(["unknown"])

    # Try C++ engine first
    if _CPP_AVAILABLE and os.path.exists(SCRIPTED_MODEL_PATH):
        try:
            cpp_engine = _si.SignInferenceEngine(SCRIPTED_MODEL_PATH)
            print("Using C++ inference engine (LibTorch)")
        except Exception as e:
            print(f"C++ engine failed: {e}")
            cpp_engine = None

    # Fall back to TorchScript in Python
    if cpp_engine is None and os.path.exists(SCRIPTED_MODEL_PATH):
        try:
            model = torch.jit.load(SCRIPTED_MODEL_PATH, map_location=device)
            model.eval()
            print("Using TorchScript Python inference")
        except Exception as e:
            print(f"TorchScript load failed: {e}")

    # Last resort: plain state dict
    if cpp_engine is None and model is None and os.path.exists(MODEL_PATH):
        try:
            m = HybridModel(258, 64, 2, len(actions)).to(device)
            m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            m.eval()
            model = m
            print("Using state dict Python inference")
        except Exception as e:
            print(f"State dict load failed: {e}")

    # Load LLM
    try:
        llm = OllamaHandler(model="llama3")
        print("LLM ready")
    except Exception as e:
        print(f"LLM failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield


app = FastAPI(title="Sign Language Brain API", lifespan=lifespan)


class SequenceInput(BaseModel):
    sequence: List[List[float]]


class TranslationInput(BaseModel):
    words: List[str]


@app.post("/predict")
def predict_sign(payload: SequenceInput):
    if cpp_engine is None and model is None:
        raise HTTPException(status_code=503, detail="No inference engine loaded")

    try:
        if cpp_engine is not None:
            # ── C++ path ───────────────────────────────────────────────────────
            result         = cpp_engine.predict(payload.sequence)
            prediction_idx = result.index
            confidence     = result.confidence
        else:
            # ── Python fallback path ───────────────────────────────────────────
            input_data = np.array(payload.sequence)
            if input_data.shape != (30, 258):
                raise HTTPException(status_code=400, detail=f"Expected shape (30, 258), got {input_data.shape}")

            tensor_data = torch.FloatTensor(input_data).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor_data)
                probs  = torch.softmax(output, dim=1).cpu().numpy()[0]

            prediction_idx = int(np.argmax(probs))
            confidence     = float(probs[prediction_idx])

        action_label = str(actions[prediction_idx]) if prediction_idx < len(actions) else "Error"

        if confidence > CONFIDENCE_THRESHOLD:
            return {"action": action_label, "confidence": confidence, "status": "success"}
        else:
            return {"action": None, "confidence": confidence, "status": "ignored"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate")
def translate_sentence(payload: TranslationInput):
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    try:
        return {"sentence": llm.process_words(payload.words)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
