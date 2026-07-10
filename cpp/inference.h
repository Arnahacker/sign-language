#pragma once
#include <torch/script.h>
#include <string>
#include <vector>

struct Prediction {
    int   index;
    float confidence;
};

class SignInferenceEngine {
public:
    explicit SignInferenceEngine(const std::string& model_path);
    Prediction predict(const std::vector<std::vector<float>>& sequence);

private:
    torch::jit::script::Module model_;
    torch::Device              device_;
};
