#include "inference.h"
#include <stdexcept>

SignInferenceEngine::SignInferenceEngine(const std::string& model_path)
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
    try {
        model_ = torch::jit::load(model_path, device_);
        model_.eval();
    } catch (const c10::Error& e) {
        throw std::runtime_error("Failed to load TorchScript model: " + std::string(e.what()));
    }
}

Prediction SignInferenceEngine::predict(const std::vector<std::vector<float>>& sequence)
{
    if (sequence.size() != 30) {
        throw std::invalid_argument(
            "Expected 30 frames, got " + std::to_string(sequence.size())
        );
    }

    // Flatten (30, 258) into a contiguous buffer for from_blob
    std::vector<float> flat;
    flat.reserve(30 * 258);

    for (const auto& frame : sequence) {
        if (frame.size() != 258) {
            throw std::invalid_argument(
                "Expected 258 features per frame, got " + std::to_string(frame.size())
            );
        }
        flat.insert(flat.end(), frame.begin(), frame.end());
    }

    // Build tensor: from_blob shares memory, so .clone() before moving to device
    auto tensor = torch::from_blob(flat.data(), {1, 30, 258})
                      .clone()
                      .to(device_);

    torch::NoGradGuard no_grad;
    auto output = model_.forward({tensor}).toTensor();
    auto probs  = torch::softmax(output, /*dim=*/1);

    auto max_result = probs.max(/*dim=*/1);
    int   idx        = std::get<1>(max_result).item<int>();
    float confidence = std::get<0>(max_result).item<float>();

    return {idx, confidence};
}
