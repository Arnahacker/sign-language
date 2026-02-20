from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import os
from typing import List, Optional
from Converting_words_to_sentences import OllamaHandler 

app = FastAPI(title="Sign Language Brain API")
CONFIDENCE_THRESHOLD = 0.85
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Final_model_parameters", "action.pth")
LABELS_PATH = os.path.join(BASE_DIR, "Final_model_parameters", "labels.npy")

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

model = None
actions = None
llm = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.on_event("startup")
def load_resources():
    global model, actions, llm
    print("API Starting up")
    
    if os.path.exists(LABELS_PATH):
        actions = np.load(LABELS_PATH)
        print(f"Loaded Labels: {LABELS_PATH}")
    else:
        print(f"Error: Labels not found at {LABELS_PATH}")
        actions = np.array(["unknown"]) 

    input_size = 258
    hidden_size = 64
    num_layers = 2
    num_classes = len(actions)

    model = HybridModel(input_size, hidden_size, num_layers, num_classes).to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Loaded Model: {MODEL_PATH}")
        except Exception as e:
            print(f"Œ Model Load Failed: {e}")
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
    
    # Load LLM
    try:
        llm = OllamaHandler(model="llama3") 
        print("LLM Ready")
    except Exception as e:
        print(f"LLM Failed: {e}")

# --- 5. DATA STRUCTURES ---
class SequenceInput(BaseModel):
    sequence: List[List[float]] 

class TranslationInput(BaseModel):
    words: List[str]

# --- 6. ENDPOINTS ---

@app.post("/predict")
def predict_sign(payload: SequenceInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_data = np.array(payload.sequence)
        
        # Validation
        if input_data.shape != (30, 258):
            raise HTTPException(status_code=400, detail=f"Expected shape (30, 258), got {input_data.shape}")

        tensor_data = torch.FloatTensor(input_data).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor_data)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        prediction_idx = np.argmax(probs)
        confidence = float(probs[prediction_idx])
        
        if prediction_idx < len(actions):
            action_label = str(actions[prediction_idx])
        else:
            action_label = "Error"

        if confidence > CONFIDENCE_THRESHOLD:
            return {
                "action": action_label,     
                "confidence": confidence,   
                "status": "success"
            }
        else:
            return {
                "action": None,             
                "confidence": confidence,   
                "status": "ignored"         
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
def translate_sentence(payload: TranslationInput):
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    try:
        refined = llm.process_words(payload.words)
        return {"sentence": refined}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))