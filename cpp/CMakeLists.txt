cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sign_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── LibTorch ───────────────────────────────────────────────────────────────────
# Pass via: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)

# ── pybind11 (installed via pip) ───────────────────────────────────────────────
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_EXIT
)

if(NOT pybind11_EXIT EQUAL 0)
    message(FATAL_ERROR "pybind11 not found. Run: pip install pybind11")
endif()

find_package(pybind11 CONFIG REQUIRED)

# ── Build the extension module ─────────────────────────────────────────────────
pybind11_add_module(sign_inference
    inference.cpp
    bindings.cpp
)

target_link_libraries(sign_inference PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET sign_inference PROPERTY CXX_STANDARD 17)
