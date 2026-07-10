# Sign Language Translation System

A real-time American Sign Language (ASL) recognition and translation pipeline that converts live hand gestures into spoken English sentences. Built with a Bidirectional LSTM-Attention model, a C++ inference engine via LibTorch and pybind11, an LLM-powered sentence construction layer, and a FastAPI backend — containerised with Docker and tested via GitHub Actions CI/CD.

---

## What It Does

The system takes a live webcam feed, detects hand and body landmarks, classifies the gesture being performed, assembles recognised words into grammatically correct sentences using a local LLM, and speaks them aloud — all in real time at 30 FPS.

The full pipeline is:

```
Camera → MediaPipe landmark extraction → C++ inference (LibTorch) → word buffer → LLaMA 3 sentence construction → TTS output
```

---

## Project Structure

```
sign_language_app/
├── cpp/                          # C++ inference engine
│   ├── inference.h               # SignInferenceEngine class declaration
│   ├── inference.cpp             # LibTorch model loading and inference logic
│   ├── bindings.cpp              # pybind11 bindings exposing C++ to Python
│   └── CMakeLists.txt            # CMake build configuration
├── tests/
│   └── smoke_test.py             # CI smoke test — no trained model required
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions workflow
├── Final_model_parameters/
│   ├── action.pth                # Trained model weights (state dict)
│   ├── action_scripted.pt        # TorchScript export (used by C++ engine)
│   └── labels.npy                # Gesture class labels
├── model.py                      # Shared HybridModel architecture definition
├── api.py                        # FastAPI inference server
├── training_code.py              # Model training + TorchScript export
├── export_torchscript.py         # Standalone TorchScript export script
├── data_collection.py            # Webcam data collection tool
├── converting_words_to_sentences.py   # Ollama/LLaMA 3 gloss-to-sentence handler
├── converting_sentences_to_speech.py  # pyttsx3 text-to-speech output
├── build.sh                      # One-command C++ build script
├── Dockerfile                    # Multi-stage Docker build
├── .dockerignore
└── requirements.txt
```

---

## Architecture Decisions

### Why Bidirectional LSTM with Attention?

Sign language gestures are temporal sequences — the meaning of a gesture depends on how it evolves across 30 frames, not just any single frame. A Bidirectional LSTM processes each sequence in both forward and backward directions, capturing context from both ends of the gesture. The attention mechanism then learns which frames within the sequence are most important for classification, rather than treating all 30 frames equally.

A 1D CNN was considered as an alternative (faster, more parallelisable) but rejected because it uses fixed-size local windows and loses long-range temporal context — a significant weakness for short 30-frame gesture clips where the relationship between the start and end of a motion carries meaning.

### Why MediaPipe and 258-Dimensional Keypoints?

MediaPipe Holistic provides skeletal landmark tracking across the body, left hand, and right hand without requiring a GPU for detection. The keypoint vector is constructed as:

- **Pose** (body): 33 landmarks × 4 values (x, y, z, visibility) = 132
- **Left hand**: 21 landmarks × 3 values (x, y, z) = 63
- **Right hand**: 21 landmarks × 3 values (x, y, z) = 63

Total: **258 dimensions per frame**, 30 frames per sequence = a (30, 258) input tensor per prediction.

Face landmarks were intentionally excluded — ASL is primarily expressed through hands and body, and including face data added noise without improving accuracy.

### Why TorchScript and LibTorch?

The original Python inference path requires the PyTorch Python runtime, which adds overhead from the Python interpreter and the Python-to-C++ bridge on every forward pass. By exporting the trained model to TorchScript with `torch.jit.trace()`, the model is compiled into a platform-independent intermediate representation that can be loaded and executed directly in C++ via LibTorch — bypassing Python entirely for inference.

The model uses `trace` rather than `script` because the forward pass has no dynamic control flow (no Python `if` statements or loops that depend on tensor values), making tracing simpler and equally correct.

### Why pybind11?

The C++ inference engine needs to be called from the Python FastAPI server. Rather than spawning a separate process and using inter-process communication, pybind11 compiles the C++ code into a native Python extension module (`.so` on Linux/Mac). The result is a zero-overhead function call from Python into C++ — no serialisation, no sockets, no subprocess.

### Why a Local LLM (Ollama + LLaMA 3)?

The gesture classifier produces individual word-level outputs, known in sign language as "glosses" — for example: `["HELLO", "NAME", "WHAT"]`. These are not grammatical English sentences. A rule-based approach to converting glosses to sentences would require hand-crafting grammar rules for every possible combination, which does not scale.

Instead, the recognised words are sent to a locally-running LLaMA 3 instance via Ollama with a system prompt that instructs it to act as a sign language interpreter. The model converts the gloss sequence into natural spoken English in a single inference call. Running it locally means no API costs, no internet dependency, and no latency from external API calls.

A sliding context window of the last 10 conversation turns is maintained, so the LLM can handle follow-up sentences with pronoun resolution and context continuity.

### Why FastAPI?

FastAPI gives automatic request validation via Pydantic models, automatic API documentation at `/docs`, and async support — all with minimal boilerplate. The confidence threshold logic (ignoring predictions below 0.85) is enforced at the API layer, keeping the client simple.

The deprecated `@app.on_event("startup")` pattern was replaced with FastAPI's `lifespan` context manager, which is the correct approach from FastAPI 0.93 onwards and avoids deprecation warnings on modern versions.

---

## Data

The dataset combines:
- **Kaggle ASL dataset** as the base (pre-recorded gesture sequences)
- **Self-recorded gestures** added via `data_collection.py` to improve coverage and adapt to the specific recording setup

Total: **35,000+ frames across 28 ASL gesture classes**

### Data Augmentation

To prevent overfitting on a relatively small dataset, each original sequence generates two additional augmented variants during training:

- **Gaussian noise** (`μ=0, σ=0.02`) added to all keypoint values — simulates natural hand tremor and minor tracking inaccuracies
- **Scale jitter** (uniform scaling between 0.95–1.05) — simulates slight differences in how far the signer's hands are from the camera

This triples the effective dataset size without collecting additional data.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Test accuracy | 82.78% |
| Macro F1 score | 0.805 |
| Gesture classes | 28 |
| Inference speed | 30 FPS |
| Input shape | (30, 258) |
| Training epochs | 80 |

---

## Problems Faced

### 1. Duplicate Code Across Three Files

`HybridModel` was originally defined separately in `training_code.py`, `api.py`, and `client_camera.py`. This meant that any change to the model architecture (adding a layer, changing hidden size) had to be made in three places, and inconsistencies would cause model load failures at inference time.

**Solution**: extracted `HybridModel` into a single `model.py` imported everywhere.

### 2. `client_camera.py` Was an Exact Duplicate of `api.py`

The file served no distinct purpose and would silently diverge over time as changes were made to one but not the other.

**Solution**: deleted `client_camera.py` entirely.

### 3. Case-Sensitive Import Failure on Linux

`Converting_words_to_sentences.py` (PascalCase) imported from `converting_sentences_to_speech.py` (snake_case). On macOS (case-insensitive filesystem) this works. On Linux (case-sensitive filesystem, i.e., any production server or Docker container) this causes an `ImportError` at runtime.

**Solution**: renamed all files to consistent snake_case.

### 4. Deprecated FastAPI Startup Event

`@app.on_event("startup")` was removed in FastAPI 0.109. Using it on modern FastAPI versions raises a deprecation warning and will break in future versions.

**Solution**: replaced with the `lifespan` context manager pattern.

### 5. `except: pass` Silently Hiding Corrupt Data

The original training loop used a bare `except: pass` when loading sequence files, meaning corrupt `.npy` files, missing frames, or wrong shapes were silently skipped with no indication of which sequences were bad or how many were lost.

**Solution**: replaced with `except Exception as e: print(f"Skipping {action}/{sequence}: {e}")` so data loading issues are visible.

### 6. TTS Blocking the Main Thread

`pyttsx3` initialised its audio engine on the main thread, which caused the video feed to freeze for the duration of speech output — a serious problem for a real-time system.

**Solution**: speech is run in a daemon thread, so the main inference loop continues uninterrupted while audio plays in the background.

### 7. TorchScript Tracing vs. Scripting

`torch.jit.script()` (the default recommendation) failed on the model because `nn.Sequential` with `BatchNorm1d` produces intermediate traced operations that the TorchScript compiler struggles to resolve in some configurations.

**Solution**: used `torch.jit.trace()` instead, which records the operations from an actual forward pass rather than analysing the Python source. Valid here because the model has no data-dependent control flow.

---

## Setup

### Requirements

- Python 3.10+
- CMake 3.18+
- A C++17 compiler (GCC or Clang)
- [Ollama](https://ollama.com) running locally with LLaMA 3 pulled

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python training_code.py
```

This trains, evaluates, saves weights to `Final_model_parameters/action.pth`, and automatically exports the TorchScript model to `Final_model_parameters/action_scripted.pt`.

### Build the C++ inference engine

Download LibTorch from [pytorch.org](https://pytorch.org/get-started/locally/) (select LibTorch, C++, your platform) and unzip as `libtorch/` in the project root, then:

```bash
chmod +x build.sh
./build.sh
```

This compiles the C++ extension and places `sign_inference.so` in the project root.

### Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

On startup the API logs which inference engine it loaded:
- `Using C++ inference engine (LibTorch)` — C++ path, fastest
- `Using TorchScript Python inference` — Python fallback if C++ module not built
- `Using state dict Python inference` — last resort if TorchScript export not run

### Run with Docker

```bash
docker build -t sign-language-api .
docker run -p 8000:8000 sign-language-api
```

The multi-stage Dockerfile compiles the C++ extension in Stage 1 and copies only the compiled `.so` into the lean Stage 2 runtime image.

### Collect new gesture data

```bash
python data_collection.py
```

Enter a word when prompted, press **Space** to record a 30-frame take, press **Q** to move to the next word.

---

## API Endpoints

### `POST /predict`

Accepts a 30-frame keypoint sequence and returns the predicted gesture.

```json
{
  "sequence": [[...258 floats...], ...]
}
```

Response:
```json
{
  "action": "HELLO",
  "confidence": 0.94,
  "status": "success"
}
```

Predictions below the confidence threshold (0.85) return `"status": "ignored"` and `"action": null`.

### `POST /translate`

Accepts a list of recognised words and returns a grammatically correct English sentence.

```json
{
  "words": ["HELLO", "NAME", "WHAT"]
}
```

Response:
```json
{
  "sentence": "Hello, what is your name?"
}
```

---

## CI/CD

Every push to `main` triggers the GitHub Actions workflow which:

1. Installs Python dependencies including PyTorch (CPU build)
2. Installs CMake and build tools
3. Compiles the C++ extension against the installed PyTorch LibTorch headers
4. Runs `tests/smoke_test.py` which creates a dummy model, exports it to TorchScript, and verifies both the C++ engine and Python inference paths produce valid outputs

The smoke test does not require the trained model — it generates a small dummy model at test time, so CI passes cleanly on a fresh checkout.
