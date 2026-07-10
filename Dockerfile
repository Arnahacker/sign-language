# ── Stage 1: Build C++ extension ───────────────────────────────────────────────
FROM python:3.10 AS builder

RUN apt-get update && apt-get install -y \
    cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install torch (CPU build) for LibTorch cmake files + pybind11
RUN pip install --no-cache-dir \
    "torch==2.1.0" --index-url https://download.pytorch.org/whl/cpu \
    pybind11

WORKDIR /build
COPY cpp/ ./cpp/

RUN TORCH_CMAKE=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    mkdir build && cd build && \
    cmake ../cpp \
        -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release -j$(nproc)

# ── Stage 2: Runtime ───────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Pull compiled .so from builder
COPY --from=builder /build/build/sign_inference*.so ./

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source files
COPY model.py \
     api.py \
     converting_words_to_sentences.py \
     converting_sentences_to_speech.py ./

# Model weights — mount a volume in production, copy for portability
COPY Final_model_parameters/ ./Final_model_parameters/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
