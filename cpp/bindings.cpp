#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "inference.h"

namespace py = pybind11;

PYBIND11_MODULE(sign_inference, m) {
    m.doc() = "Sign language C++ inference engine via LibTorch";

    py::class_<Prediction>(m, "Prediction")
        .def_readonly("index",      &Prediction::index)
        .def_readonly("confidence", &Prediction::confidence);

    py::class_<SignInferenceEngine>(m, "SignInferenceEngine")
        .def(py::init<const std::string&>(), py::arg("model_path"),
             "Load a TorchScript model from disk")
        .def("predict", &SignInferenceEngine::predict, py::arg("sequence"),
             "Run inference on a (30, 258) sequence. Returns a Prediction.");
}
